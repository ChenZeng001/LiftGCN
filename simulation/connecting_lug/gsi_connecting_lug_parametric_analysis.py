# -*- coding: utf-8 -*-
"""Robust parametric 3-D connecting lug analysis for Abaqus/CAE.

This file is a root-cause-fixed version of the supplied connecting-lug script.

Key fixes
---------
1. Fixes the Abaqus Job-status race:
   Immediately after Job.submit(), job.status can still be NONE while the
   message queue is empty. Abaqus waitForCompletion() returns immediately when
   the status is neither SUBMITTED nor RUNNING, which can make a valid analysis
   look like a failure with "status: None".

   This script therefore synchronizes submission explicitly and uses both the
   Job API and solver files (.lck/.sta/.log/.msg/.dat) to decide whether the
   solver has actually started, completed, or failed.

2. Strengthens C3D8R/C3D20R meshing:
   The part is an exact straight extrusion, so pure HEX meshes now prefer a
   bottom-up extrusion of a pure-QUAD source mesh. The target extrusion face is
   passed explicitly, so the final layer terminates on the real target face.
   Top-down sweep remains as a fallback.

3. Adds Abaqus ANALYSIS_CHECKS mesh verification:
   A mesh strategy is rejected before job submission if Abaqus reports failed
   elements through verifyMeshQuality(ANALYSIS_CHECKS). Older Abaqus releases
   that do not expose this API continue with a warning.

4. Keeps detailed solver diagnostics:
   On a real solver failure, the temporary run directory is retained and the
   tails of .log/.sta/.msg/.dat/.prt are included in the exception text.

SI units are used (m, Pa, N).

Geometric/load parameter order:
    shank_length, outer_radius, pin_hole_radius, mounting_hole_radius,
    thickness, mesh_size, force_magnitude

Supported 3-D element types:
    C3D4, C3D10M, C3D6, C3D8R, C3D20R

Default element type:
    C3D10M

Examples
--------
Default:
    abaqus cae noGUI=gsi_connecting_lug_parametric_analysis_fixed_all_elements.py

One type:
    abaqus cae noGUI=gsi_connecting_lug_parametric_analysis_fixed_all_elements.py -- C3D8R

Seven parameters plus one type:
    abaqus cae noGUI=gsi_connecting_lug_parametric_analysis_fixed_all_elements.py -- \
        0.080 0.030 0.015 0.005 0.020 0.007 30000 C3D8R

All types:
    abaqus cae noGUI=gsi_connecting_lug_parametric_analysis_fixed_all_elements.py -- ALL

Environment variable:
    CONNECTING_LUG_PARAMS

Optional synchronization environment variables:
    CONNECTING_LUG_JOB_START_TIMEOUT
        Seconds allowed for Abaqus to show solver activity. Default: 180.

    CONNECTING_LUG_JOB_NOLOCK_GRACE
        Grace period after solver activity is seen but no .lck file/status is
        active before declaring a terminal failure. Default: 10.

Important Abaqus API container rule
-----------------------------------
Keep these types separate:
    * setSweepPath(edge=...) -> one Edge.
    * setMeshControls(regions=...) -> FaceArray/CellArray sequence.
    * regionToolset.Region(faces=...) -> FaceArray/GeomSequence.
    * generateBottomUpExtrudedMesh(geometrySourceSide=...) -> Region.
"""

from __future__ import print_function

import math
import os
import shutil
import sys
import tempfile
import time

from PIL import Image
from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import regionToolset

DEFAULTS = {
    'shank_length': 0.080,
    'outer_radius': 0.030,
    'pin_hole_radius': 0.015,
    'mounting_hole_radius': 0.005,
    'thickness': 0.020,
    'mesh_size': 0.007,
    'force_magnitude': 30000.0,
}

PARAMETER_NAMES = (
    'shank_length', 'outer_radius', 'pin_hole_radius',
    'mounting_hole_radius', 'thickness', 'mesh_size', 'force_magnitude'
)

SUPPORTED_ELEMENT_TYPES = (
    'C3D4',
    'C3D10M',
    'C3D6',
    'C3D8R',
    'C3D20R',
)
DEFAULT_ELEMENT_TYPE = 'C3D10M'
ALL_ELEMENT_TYPES_KEYWORD = 'ALL'

MODEL_NAME = 'ConnectingLugModel'
PART_NAME = 'ConnectingLug'
INSTANCE_NAME = 'ConnectingLug-1'
STEP_NAME = 'LugLoad'
JOB_NAME = 'ConnectingLugAnalysis'

YOUNGS_MODULUS = 200.0E9
POISSONS_RATIO = 0.30

JOB_POLL_INTERVAL_SECONDS = 0.25
DEFAULT_JOB_START_TIMEOUT_SECONDS = 180.0
DEFAULT_JOB_NOLOCK_GRACE_SECONDS = 10.0
DIAGNOSTIC_TAIL_BYTES = 12000

# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def script_directory():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def environment_float(name, default_value):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return float(default_value)

    value = float(raw)
    if value <= 0.0:
        raise ValueError('%s must be greater than zero.' % name)
    return value

def normalize_element_selector(value):
    selector = str(value).strip().upper()
    valid = SUPPORTED_ELEMENT_TYPES + (ALL_ELEMENT_TYPES_KEYWORD,)

    if selector not in valid:
        raise ValueError(
            'Unsupported element type "%s". Choose one of: %s, or ALL.'
            % (value, ', '.join(SUPPORTED_ELEMENT_TYPES))
        )

    return selector

def command_line_parameters():
    """Return (parameter_dictionary, element_type_list)."""
    raw = os.environ.get('CONNECTING_LUG_PARAMS', '').strip()

    if raw:
        args = raw.replace(',', ' ').split()
    else:
        args = list(sys.argv[1:])
        if '--' in args:
            args = args[args.index('--') + 1:]

    selector = DEFAULT_ELEMENT_TYPE

    if not args:
        values = dict(DEFAULTS)

    elif len(args) == 1:
        selector = normalize_element_selector(args[0])
        values = dict(DEFAULTS)

    elif len(args) == len(PARAMETER_NAMES):
        values = {}
        for name, value in zip(PARAMETER_NAMES, args):
            values[name] = float(value)

    elif len(args) == len(PARAMETER_NAMES) + 1:
        values = {}
        for name, value in zip(
            PARAMETER_NAMES,
            args[:len(PARAMETER_NAMES)]
        ):
            values[name] = float(value)

        selector = normalize_element_selector(args[-1])

    else:
        raise ValueError(
            'Expected one of the following argument forms:\n'
            '  no arguments\n'
            '  ELEMENT_TYPE\n'
            '  seven values: %s\n'
            '  seven values followed by ELEMENT_TYPE\n'
            'ELEMENT_TYPE must be one of %s or ALL.'
            % (
                ' '.join(PARAMETER_NAMES),
                ', '.join(SUPPORTED_ELEMENT_TYPES),
            )
        )

    validate_parameters(values)

    if selector == ALL_ELEMENT_TYPES_KEYWORD:
        element_types = list(SUPPORTED_ELEMENT_TYPES)
    else:
        element_types = [selector]

    return values, element_types

def validate_parameters(p):
    for name in PARAMETER_NAMES:
        if p[name] <= 0.0:
            raise ValueError('%s must be greater than zero.' % name)

    radius = p['outer_radius']

    if p['shank_length'] <= 1.8 * radius:
        raise ValueError(
            'shank_length must be greater than 1.8*outer_radius.'
        )

    if p['pin_hole_radius'] >= 0.62 * radius:
        raise ValueError(
            'pin_hole_radius must be smaller than 0.62*outer_radius.'
        )

    if p['mounting_hole_radius'] >= 0.23 * radius:
        raise ValueError(
            'mounting_hole_radius must be smaller than 0.23*outer_radius.'
        )

    if p['mesh_size'] >= min(p['thickness'], radius):
        raise ValueError(
            'mesh_size must be smaller than thickness and outer_radius.'
        )

def make_viewport():
    name = 'Viewport: 1'

    if name not in session.viewports.keys():
        session.Viewport(
            name=name,
            origin=(0.0, 0.0),
            width=200.0,
            height=150.0,
        )

    viewport = session.viewports[name]
    viewport.makeCurrent()

    try:
        viewport.maximize()
    except Exception:
        pass

    return viewport

def print_viewport_to_jpg(viewport, temporary_base, jpg_path):
    png_path = temporary_base + '.png'

    session.printToFile(
        fileName=temporary_base,
        format=PNG,
        canvasObjects=(viewport,),
    )

    rendered = Image.open(png_path)
    try:
        rendered.convert('RGB').save(
            jpg_path,
            format='JPEG',
            quality=95,
        )
    finally:
        rendered.close()

    os.remove(png_path)

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def create_geometry(model, p):
    length = p['shank_length']
    radius = p['outer_radius']
    sheet_size = 4.0 * (length + radius)

    sketch = model.ConstrainedSketch(
        name='__profile__',
        sheetSize=sheet_size,
    )

    sketch.Line(
        point1=(0.0, -radius),
        point2=(length, -radius),
    )

    sketch.ArcByCenterEnds(
        center=(length, 0.0),
        point1=(length, -radius),
        point2=(length, radius),
        direction=COUNTERCLOCKWISE,
    )

    sketch.Line(
        point1=(length, radius),
        point2=(0.0, radius),
    )

    sketch.Line(
        point1=(0.0, radius),
        point2=(0.0, -radius),
    )

    pin_radius = p['pin_hole_radius']
    sketch.CircleByCenterPerimeter(
        center=(length, 0.0),
        point1=(length + pin_radius, 0.0),
    )

    mounting_x = 0.30 * length
    mounting_y = 0.42 * radius
    mounting_radius = p['mounting_hole_radius']

    for sign in (-1.0, 1.0):
        center = (mounting_x, sign * mounting_y)
        sketch.CircleByCenterPerimeter(
            center=center,
            point1=(center[0] + mounting_radius, center[1]),
        )

    part = model.Part(
        name=PART_NAME,
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY,
    )

    part.BaseSolidExtrude(
        sketch=sketch,
        depth=p['thickness'],
    )

    del model.sketches['__profile__']
    return part

def pin_load_node_labels(instance, p):
    """Return mesh nodes on the lower half of the main cylindrical hole."""
    center_x = p['shank_length']
    hole_radius = p['pin_hole_radius']

    radial_tolerance = max(
        1.0E-8,
        0.04 * p['mesh_size'],
    )

    y_tolerance = max(
        1.0E-10,
        1.0E-4 * p['mesh_size'],
    )

    labels = []

    for node in instance.nodes:
        x, y, unused_z = node.coordinates
        del unused_z

        radial_distance = math.sqrt(
            (x - center_x) ** 2 + y ** 2
        )

        if (
            abs(radial_distance - hole_radius) <= radial_tolerance
            and y <= y_tolerance
        ):
            labels.append(int(node.label))

    return sorted(set(labels))

# ---------------------------------------------------------------------------
# Element-type and geometry-selection helpers
# ---------------------------------------------------------------------------

def element_code_constant(element_type):
    if element_type == 'C3D4':
        return C3D4
    if element_type == 'C3D10M':
        return C3D10M
    if element_type == 'C3D6':
        return C3D6
    if element_type == 'C3D8R':
        return C3D8R
    if element_type == 'C3D20R':
        return C3D20R

    raise ValueError(
        'Unsupported element type: %s'
        % element_type
    )

def extrusion_probe_point(p, z_value):
    return (
        0.50 * p['shank_length'],
        0.0,
        z_value,
    )

def extrusion_end_face_objects(part, p):
    """Return exactly one source Face and one target Face."""
    source_face = part.faces.findAt(
        coordinates=extrusion_probe_point(p, 0.0)
    )

    target_face = part.faces.findAt(
        coordinates=extrusion_probe_point(
            p,
            p['thickness'],
        )
    )

    return source_face, target_face

def extrusion_end_face_sequences(part, p):
    """Return one-face FaceArray/GeomSequence source/target selections."""
    source_faces = part.faces.findAt(
        (extrusion_probe_point(p, 0.0),)
    )

    target_faces = part.faces.findAt(
        (
            extrusion_probe_point(
                p,
                p['thickness'],
            ),
        )
    )

    if len(source_faces) != 1 or len(target_faces) != 1:
        raise RuntimeError(
            'Expected exactly one source and one target extrusion face; '
            'found %d and %d.'
            % (
                len(source_faces),
                len(target_faces),
            )
        )

    return source_faces, target_faces

def extrusion_sweep_edge(part, p):
    """Return one Edge object parallel to the extrusion direction."""
    return part.edges.findAt(
        coordinates=(
            0.0,
            -p['outer_radius'],
            0.5 * p['thickness'],
        )
    )

def extrusion_sweep_edge_sequence(part, p):
    """Return a one-edge EdgeArray/GeomSequence for seeding."""
    edges = part.edges.findAt(
        (
            (
                0.0,
                -p['outer_radius'],
                0.5 * p['thickness'],
            ),
        )
    )

    if len(edges) != 1:
        raise RuntimeError(
            'Expected exactly one extrusion-direction edge; found %d.'
            % len(edges)
        )

    return edges

def source_face_region(part, p):
    """Return Region(FaceArray) for bottom-up geometrySourceSide."""
    source_faces, unused_target_faces = extrusion_end_face_sequences(
        part,
        p,
    )
    del unused_target_faces

    return regionToolset.Region(
        faces=source_faces,
    )

def whole_cell_region(part):
    return regionToolset.Region(
        cells=part.cells[:],
    )

def sweep_layer_count(p):
    """Keep the layer thickness no larger than the requested mesh size."""
    return max(
        1,
        int(
            math.ceil(
                p['thickness'] / p['mesh_size']
            )
        ),
    )

# ---------------------------------------------------------------------------
# Mesh verification
# ---------------------------------------------------------------------------

def mesh_is_complete(part):
    if len(part.elements) == 0:
        return False

    try:
        return part.getUnmeshedRegions() is None
    except Exception:
        return len(part.elements) > 0

def mesh_element_types(part):
    result = set()

    for element in part.elements:
        result.add(
            str(element.type).upper()
        )

    return sorted(result)

def verify_mesh_analysis_checks(part, requested_type):
    """Reject elements that fail Abaqus input-processor quality checks.

    verifyMeshQuality(ANALYSIS_CHECKS) exists in current Abaqus releases.
    If an older release does not expose it, retain backward compatibility and
    continue after printing a warning.
    """
    try:
        result = part.verifyMeshQuality(
            criterion=ANALYSIS_CHECKS,
        )
    except Exception as exc:
        print(
            'Warning: ANALYSIS_CHECKS mesh verification unavailable: %s'
            % exc
        )
        return

    failed = result.get(
        'failedElements',
        (),
    )
    warnings = result.get(
        'warningElements',
        (),
    )
    na_elements = result.get(
        'naElements',
        (),
    )

    try:
        failed_count = len(failed)
    except Exception:
        failed_count = 0

    try:
        warning_count = len(warnings)
    except Exception:
        warning_count = 0

    try:
        na_count = len(na_elements)
    except Exception:
        na_count = 0

    if failed_count:
        labels = []
        for element in failed[:20]:
            try:
                labels.append(
                    str(int(element.label))
                )
            except Exception:
                pass

        label_text = (
            ', '.join(labels)
            if labels
            else '<labels unavailable>'
        )

        raise RuntimeError(
            'Abaqus ANALYSIS_CHECKS rejected %d %s element(s). '
            'First failed labels: %s'
            % (
                failed_count,
                requested_type,
                label_text,
            )
        )

    print(
        'Mesh analysis checks: failed=%d warning=%d n/a=%d'
        % (
            failed_count,
            warning_count,
            na_count,
        )
    )

def verify_requested_element_type(part, requested_type):
    if not mesh_is_complete(part):
        raise RuntimeError(
            'Mesh generation did not completely mesh the 3-D solid.'
        )

    generated_types = mesh_element_types(
        part
    )

    if generated_types != [requested_type]:
        raise RuntimeError(
            'Requested a pure %s mesh, but Abaqus generated '
            'element type(s): %s'
            % (
                requested_type,
                ', '.join(generated_types)
                if generated_types
                else '<none>',
            )
        )

    verify_mesh_analysis_checks(
        part,
        requested_type,
    )

    print(
        'Mesh generated: %d nodes, %d elements, type=%s'
        % (
            len(part.nodes),
            len(part.elements),
            requested_type,
        )
    )

def delete_existing_mesh(part):
    """Delete full/partial/native mesh and any boundary preview safely."""
    try:
        part.deletePreviewMesh()
    except Exception:
        pass

    try:
        part.deleteMesh(
            regions=(part,),
        )
    except Exception:
        try:
            region = whole_cell_region(part)
            part.deleteMesh(
                regions=(region,),
            )
        except Exception:
            pass

def reseed_part(part, p):
    part.seedPart(
        size=p['mesh_size'],
        deviationFactor=0.1,
        minSizeFactor=0.1,
    )

def assign_solid_element_type(part, element_type):
    elem_type = mesh.ElemType(
        elemCode=element_code_constant(
            element_type
        ),
        elemLibrary=STANDARD,
    )

    part.setElementType(
        regions=(part.cells[:],),
        elemTypes=(elem_type,),
    )

def print_mesh_controls(part, label):
    try:
        cell = part.cells[0]

        shape = part.getMeshControl(
            region=cell,
            attribute=ELEM_SHAPE,
        )

        technique = part.getMeshControl(
            region=cell,
            attribute=TECHNIQUE,
        )

        algorithm = part.getMeshControl(
            region=cell,
            attribute=ALGORITHM,
        )

        print(
            '%s controls: shape=%s technique=%s algorithm=%s'
            % (
                label,
                shape,
                technique,
                algorithm,
            )
        )

    except Exception as exc:
        print(
            'Warning: could not query %s mesh controls: %s'
            % (
                label,
                exc,
            )
        )

# ---------------------------------------------------------------------------
# Tetrahedral mesh
# ---------------------------------------------------------------------------

def configure_tet_mesh(part, p, element_type):
    delete_existing_mesh(part)
    reseed_part(part, p)

    part.setMeshControls(
        regions=part.cells[:],
        elemShape=TET,
        technique=FREE,
    )

    assign_solid_element_type(
        part,
        element_type,
    )

    part.generateMesh()

    verify_requested_element_type(
        part,
        element_type,
    )

# ---------------------------------------------------------------------------
# Bottom-up extrusion
# ---------------------------------------------------------------------------

def apply_bottom_up_source_controls(
    part,
    p,
    volume_shape,
    source_shape,
    algorithm=None,
):
    source_faces, unused_target_faces = extrusion_end_face_sequences(
        part,
        p,
    )
    del unused_target_faces

    part.setMeshControls(
        regions=part.cells[:],
        elemShape=volume_shape,
        technique=BOTTOM_UP,
    )

    if source_shape == TRI:
        part.setMeshControls(
            regions=source_faces,
            elemShape=TRI,
            technique=FREE,
            allowMapped=OFF,
        )

    elif source_shape == QUAD:
        if algorithm == MEDIAL_AXIS:
            part.setMeshControls(
                regions=source_faces,
                elemShape=QUAD,
                technique=FREE,
                algorithm=MEDIAL_AXIS,
                minTransition=ON,
            )

        elif algorithm == ADVANCING_FRONT:
            part.setMeshControls(
                regions=source_faces,
                elemShape=QUAD,
                technique=FREE,
                algorithm=ADVANCING_FRONT,
                allowMapped=ON,
            )

        else:
            raise ValueError(
                'QUAD bottom-up source requires MEDIAL_AXIS '
                'or ADVANCING_FRONT.'
            )

    else:
        raise ValueError(
            'Unsupported bottom-up source shape: %s'
            % source_shape
        )

def generate_bottom_up_extrusion(
    part,
    p,
    element_type,
    volume_shape,
    source_shape,
    algorithm=None,
    use_boundary_preview=False,
):
    """Generate a pure bottom-up extrusion.

    Root-fix detail:
    targetSide is specified explicitly. The part is a strict extrusion, so the
    target face is the most reliable geometric termination for the last layer.
    """
    delete_existing_mesh(part)
    reseed_part(part, p)
    assign_solid_element_type(
        part,
        element_type,
    )

    apply_bottom_up_source_controls(
        part=part,
        p=p,
        volume_shape=volume_shape,
        source_shape=source_shape,
        algorithm=algorithm,
    )

    source_region = source_face_region(
        part,
        p,
    )

    unused_source_faces, target_faces = extrusion_end_face_sequences(
        part,
        p,
    )
    del unused_source_faces

    print_mesh_controls(
        part,
        'bottom-up %s'
        % element_type,
    )

    if use_boundary_preview:
        print(
            'Generating bottom-up boundary preview before extrusion...'
        )

        part.generateMesh(
            boundaryPreview=ON,
        )

    part.generateBottomUpExtrudedMesh(
        cell=part.cells[0],
        geometrySourceSide=source_region,
        targetSide=target_faces,
        numberOfLayers=sweep_layer_count(p),
        extrudeVector=(
            (0.0, 0.0, 0.0),
            (
                0.0,
                0.0,
                p['thickness'],
            ),
        ),
        biasRatio=1.0,
    )

    verify_requested_element_type(
        part,
        element_type,
    )

# ---------------------------------------------------------------------------
# Top-down sweep fallbacks
# ---------------------------------------------------------------------------

def apply_top_down_wedge_sweep_controls(part, p):
    source_faces, unused_target_faces = extrusion_end_face_sequences(
        part,
        p,
    )
    del unused_target_faces

    sweep_edge = extrusion_sweep_edge(
        part,
        p,
    )

    sweep_edges = extrusion_sweep_edge_sequence(
        part,
        p,
    )

    part.setMeshControls(
        regions=source_faces,
        elemShape=TRI,
        technique=FREE,
        allowMapped=OFF,
    )

    part.setMeshControls(
        regions=part.cells[:],
        elemShape=WEDGE,
        technique=SWEEP,
    )

    part.setSweepPath(
        region=part.cells[0],
        edge=sweep_edge,
        sense=FORWARD,
    )

    part.seedEdgeByNumber(
        edges=sweep_edges,
        number=sweep_layer_count(p),
        constraint=FIXED,
    )

def apply_top_down_hex_sweep_controls(part, p, algorithm):
    source_faces, unused_target_faces = extrusion_end_face_sequences(
        part,
        p,
    )
    del unused_target_faces

    sweep_edge = extrusion_sweep_edge(
        part,
        p,
    )

    sweep_edges = extrusion_sweep_edge_sequence(
        part,
        p,
    )

    if algorithm == MEDIAL_AXIS:
        part.setMeshControls(
            regions=source_faces,
            elemShape=QUAD,
            technique=FREE,
            algorithm=MEDIAL_AXIS,
            minTransition=ON,
        )

        part.setMeshControls(
            regions=part.cells[:],
            elemShape=HEX,
            technique=SWEEP,
            algorithm=MEDIAL_AXIS,
            minTransition=ON,
        )

    elif algorithm == ADVANCING_FRONT:
        part.setMeshControls(
            regions=source_faces,
            elemShape=QUAD,
            technique=FREE,
            algorithm=ADVANCING_FRONT,
            allowMapped=ON,
        )

        part.setMeshControls(
            regions=part.cells[:],
            elemShape=HEX,
            technique=SWEEP,
            algorithm=ADVANCING_FRONT,
            allowMapped=ON,
        )

    else:
        raise ValueError(
            'Unsupported HEX sweep algorithm: %s'
            % algorithm
        )

    part.setSweepPath(
        region=part.cells[0],
        edge=sweep_edge,
        sense=FORWARD,
    )

    part.seedEdgeByNumber(
        edges=sweep_edges,
        number=sweep_layer_count(p),
        constraint=FIXED,
    )

def try_top_down_wedge(part, p):
    delete_existing_mesh(part)
    reseed_part(part, p)
    assign_solid_element_type(
        part,
        'C3D6',
    )

    apply_top_down_wedge_sweep_controls(
        part,
        p,
    )

    print_mesh_controls(
        part,
        'top-down C3D6',
    )

    part.generateMesh()

    verify_requested_element_type(
        part,
        'C3D6',
    )

def try_top_down_hex(
    part,
    p,
    element_type,
    algorithm,
):
    delete_existing_mesh(part)
    reseed_part(part, p)
    assign_solid_element_type(
        part,
        element_type,
    )

    apply_top_down_hex_sweep_controls(
        part,
        p,
        algorithm,
    )

    print_mesh_controls(
        part,
        'top-down %s/%s'
        % (
            element_type,
            algorithm,
        ),
    )

    part.generateMesh()

    verify_requested_element_type(
        part,
        element_type,
    )

# ---------------------------------------------------------------------------
# WEDGE/HEX strategy orchestration
# ---------------------------------------------------------------------------

def run_mesh_attempts(
    part,
    requested_type,
    attempts,
):
    errors = []

    for name, action in attempts:
        print(
            'Generating %s mesh using %s...'
            % (
                requested_type,
                name,
            )
        )

        try:
            action()

            print(
                '%s strategy succeeded: %s'
                % (
                    requested_type,
                    name,
                )
            )

            return

        except Exception as exc:
            errors.append(
                '%s: %s'
                % (
                    name,
                    exc,
                )
            )

            print(
                '%s strategy failed: %s -> %s'
                % (
                    requested_type,
                    name,
                    exc,
                )
            )

            delete_existing_mesh(
                part
            )

    raise RuntimeError(
        'Unable to generate a complete pure %s mesh. '
        'Diagnostics: %s'
        % (
            requested_type,
            ' | '.join(errors),
        )
    )

def configure_wedge_mesh(part, p):
    attempts = (
        (
            'BOTTOM_UP_TRI_DIRECT',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type='C3D6',
                volume_shape=WEDGE,
                source_shape=TRI,
                algorithm=None,
                use_boundary_preview=False,
            ),
        ),
        (
            'BOTTOM_UP_TRI_BOUNDARY_PREVIEW',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type='C3D6',
                volume_shape=WEDGE,
                source_shape=TRI,
                algorithm=None,
                use_boundary_preview=True,
            ),
        ),
        (
            'TOP_DOWN_WEDGE_SWEEP',
            lambda: try_top_down_wedge(
                part,
                p,
            ),
        ),
    )

    run_mesh_attempts(
        part,
        'C3D6',
        attempts,
    )

def configure_hex_mesh(part, p, element_type):
    """Generate a pure C3D8R/C3D20R mesh.

    Because the entire lug is a strict straight extrusion, bottom-up extrusion
    is the primary route. Top-down sweep is retained only as fallback.

    Order:
      1. bottom-up ADVANCING_FRONT direct
      2. bottom-up MEDIAL_AXIS direct
      3. bottom-up ADVANCING_FRONT + boundary preview
      4. bottom-up MEDIAL_AXIS + boundary preview
      5. top-down ADVANCING_FRONT
      6. top-down MEDIAL_AXIS
    """
    attempts = (
        (
            'BOTTOM_UP_ADVANCING_FRONT_DIRECT',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type=element_type,
                volume_shape=HEX,
                source_shape=QUAD,
                algorithm=ADVANCING_FRONT,
                use_boundary_preview=False,
            ),
        ),
        (
            'BOTTOM_UP_MEDIAL_AXIS_DIRECT',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type=element_type,
                volume_shape=HEX,
                source_shape=QUAD,
                algorithm=MEDIAL_AXIS,
                use_boundary_preview=False,
            ),
        ),
        (
            'BOTTOM_UP_ADVANCING_FRONT_BOUNDARY_PREVIEW',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type=element_type,
                volume_shape=HEX,
                source_shape=QUAD,
                algorithm=ADVANCING_FRONT,
                use_boundary_preview=True,
            ),
        ),
        (
            'BOTTOM_UP_MEDIAL_AXIS_BOUNDARY_PREVIEW',
            lambda: generate_bottom_up_extrusion(
                part=part,
                p=p,
                element_type=element_type,
                volume_shape=HEX,
                source_shape=QUAD,
                algorithm=MEDIAL_AXIS,
                use_boundary_preview=True,
            ),
        ),
        (
            'TOP_DOWN_ADVANCING_FRONT',
            lambda: try_top_down_hex(
                part,
                p,
                element_type,
                ADVANCING_FRONT,
            ),
        ),
        (
            'TOP_DOWN_MEDIAL_AXIS',
            lambda: try_top_down_hex(
                part,
                p,
                element_type,
                MEDIAL_AXIS,
            ),
        ),
    )

    run_mesh_attempts(
        part,
        element_type,
        attempts,
    )

def configure_mesh(part, p, element_type):
    print(
        'Requested element type: %s'
        % element_type
    )

    if element_type in (
        'C3D4',
        'C3D10M',
    ):
        configure_tet_mesh(
            part,
            p,
            element_type,
        )

    elif element_type == 'C3D6':
        configure_wedge_mesh(
            part,
            p,
        )

    elif element_type in (
        'C3D8R',
        'C3D20R',
    ):
        configure_hex_mesh(
            part,
            p,
            element_type,
        )

    else:
        raise ValueError(
            'Unsupported element type: %s'
            % element_type
        )

# ---------------------------------------------------------------------------
# Solver synchronization and diagnostics
# ---------------------------------------------------------------------------

def job_status_name(job):
    try:
        status = job.status
    except Exception:
        return 'UNKNOWN'

    if status is None:
        return 'NONE'

    text = str(status).strip().upper()

    if not text:
        return 'NONE'

    return text

def read_file_tail(path, max_bytes=DIAGNOSTIC_TAIL_BYTES):
    if not os.path.isfile(path):
        return ''

    handle = open(path, 'rb')

    try:
        handle.seek(
            0,
            os.SEEK_END,
        )

        size = handle.tell()
        start = max(
            0,
            size - int(max_bytes),
        )

        handle.seek(
            start,
            os.SEEK_SET,
        )

        data = handle.read()

    finally:
        handle.close()

    try:
        return data.decode(
            'utf-8',
            'replace',
        )
    except Exception:
        try:
            return data.decode(
                'latin-1',
                'replace',
            )
        except Exception:
            return str(data)

def file_contains_upper(path, token):
    if not os.path.isfile(path):
        return False

    token_upper = token.upper()

    handle = open(path, 'rb')
    try:
        data = handle.read().upper()
    finally:
        handle.close()

    if not isinstance(token_upper, bytes):
        try:
            token_upper = token_upper.encode(
                'ascii'
            )
        except Exception:
            token_upper = str(
                token_upper
            )

    return token_upper in data

def solver_completed(job, run_dir):
    if job_status_name(job) == 'COMPLETED':
        return True

    sta_path = os.path.join(
        run_dir,
        JOB_NAME + '.sta',
    )

    if file_contains_upper(
        sta_path,
        'THE ANALYSIS HAS COMPLETED SUCCESSFULLY',
    ):
        return True

    log_path = os.path.join(
        run_dir,
        JOB_NAME + '.log',
    )

    if file_contains_upper(
        log_path,
        'ABAQUS JOB %s COMPLETED'
        % JOB_NAME.upper(),
    ):
        return True

    return False

def solver_activity_paths(run_dir):
    extensions = (
        '.lck',
        '.log',
        '.sta',
        '.msg',
        '.dat',
        '.odb',
        '.com',
        '.sim',
    )

    result = []

    for extension in extensions:
        path = os.path.join(
            run_dir,
            JOB_NAME + extension,
        )

        if os.path.exists(path):
            result.append(path)

    return result

def solver_lock_exists(run_dir):
    return os.path.isfile(
        os.path.join(
            run_dir,
            JOB_NAME + '.lck',
        )
    )

def solver_failure_marker_present(run_dir):
    checks = (
        (
            '.log',
            'ABAQUS/ANALYSIS EXITED WITH ERRORS',
        ),
        (
            '.log',
            'ABAQUS JOB %s ABORTED'
            % JOB_NAME.upper(),
        ),
        (
            '.log',
            'ABAQUS JOB %s TERMINATED'
            % JOB_NAME.upper(),
        ),
        (
            '.sta',
            'THE ANALYSIS HAS NOT BEEN COMPLETED',
        ),
    )

    for extension, token in checks:
        path = os.path.join(
            run_dir,
            JOB_NAME + extension,
        )

        if file_contains_upper(
            path,
            token,
        ):
            return True

    return False

def job_message_summary(job, max_messages=20):
    lines = []

    try:
        messages = job.messages
    except Exception:
        return lines

    try:
        count = len(messages)
    except Exception:
        return lines

    start = max(
        0,
        count - int(max_messages),
    )

    for index in range(start, count):
        try:
            lines.append(
                str(messages[index])
            )
        except Exception:
            pass

    return lines

def solver_diagnostics(run_dir, job):
    lines = [
        'Job status: %s'
        % job_status_name(job),
        'Retained run directory: %s'
        % run_dir,
    ]

    messages = job_message_summary(
        job
    )

    if messages:
        lines.append(
            'Recent CAE job messages:'
        )
        lines.extend(
            [
                '  ' + message
                for message in messages
            ]
        )

    for extension in (
        '.log',
        '.sta',
        '.msg',
        '.dat',
        '.prt',
    ):
        path = os.path.join(
            run_dir,
            JOB_NAME + extension,
        )

        tail = read_file_tail(
            path
        ).strip()

        if tail:
            lines.append(
                ''
            )
            lines.append(
                '----- tail of %s -----'
                % os.path.basename(path)
            )
            lines.append(
                tail
            )

    return '\n'.join(lines)

def wait_for_abaqus_job(job, run_dir):
    """Wait robustly for Abaqus submission and completion.

    Abaqus documents that waitForCompletion() returns immediately if the job
    status is neither SUBMITTED nor RUNNING. Immediately after submit(), status
    may still be NONE because no job messages have arrived. Therefore this
    routine never calls waitForCompletion() while status is NONE.

    File-based activity monitoring is retained as a fallback for environments
    where CAE job messages are delayed or unavailable.
    """
    start_timeout = environment_float(
        'CONNECTING_LUG_JOB_START_TIMEOUT',
        DEFAULT_JOB_START_TIMEOUT_SECONDS,
    )

    no_lock_grace = environment_float(
        'CONNECTING_LUG_JOB_NOLOCK_GRACE',
        DEFAULT_JOB_NOLOCK_GRACE_SECONDS,
    )

    active_statuses = (
        'SUBMITTED',
        'RUNNING',
        'CHECK_RUNNING',
        'CHECK_SUBMITTED',
    )

    failure_statuses = (
        'ABORTED',
        'TERMINATED',
    )

    start_deadline = time.time() + start_timeout
    activity_seen = False
    no_lock_since = None
    last_reported_status = None

    while True:
        status = job_status_name(
            job
        )

        if status != last_reported_status:
            print(
                'Abaqus job status: %s'
                % status
            )
            last_reported_status = status

        if solver_completed(
            job,
            run_dir,
        ):
            return

        if status in failure_statuses:
            raise RuntimeError(
                'Abaqus solver terminated with status %s.\n%s'
                % (
                    status,
                    solver_diagnostics(
                        run_dir,
                        job,
                    ),
                )
            )

        if status in active_statuses:
            # At this point waitForCompletion() is safe: Abaqus will block
            # because the status is SUBMITTED/RUNNING rather than NONE.
            job.waitForCompletion()

            if solver_completed(
                job,
                run_dir,
            ):
                return

            status_after_wait = job_status_name(
                job
            )

            if status_after_wait in failure_statuses:
                raise RuntimeError(
                    'Abaqus solver terminated with status %s.\n%s'
                    % (
                        status_after_wait,
                        solver_diagnostics(
                            run_dir,
                            job,
                        ),
                    )
                )

            # Some installations can lose the final CAE message while the
            # solver files still correctly show completion/failure. Continue
            # through the file-based checks below.
            activity_seen = bool(
                solver_activity_paths(
                    run_dir
                )
            ) or activity_seen

        activity_paths = solver_activity_paths(
            run_dir
        )

        if activity_paths:
            activity_seen = True

        lock_exists = solver_lock_exists(
            run_dir
        )

        if lock_exists:
            no_lock_since = None

        elif activity_seen:
            if no_lock_since is None:
                no_lock_since = time.time()

            if solver_failure_marker_present(
                run_dir
            ):
                raise RuntimeError(
                    'Abaqus solver reported a failure in its output files.\n%s'
                    % solver_diagnostics(
                        run_dir,
                        job,
                    )
                )

            if (
                time.time() - no_lock_since
                >= no_lock_grace
            ):
                if solver_completed(
                    job,
                    run_dir,
                ):
                    return

                raise RuntimeError(
                    'Abaqus solver activity was detected, but the job '
                    'stopped without a successful completion marker.\n%s'
                    % solver_diagnostics(
                        run_dir,
                        job,
                    )
                )

        elif time.time() >= start_deadline:
            raise RuntimeError(
                'Abaqus job did not enter SUBMITTED/RUNNING and no solver '
                'activity files appeared within %.1f seconds. This is a '
                'submission/startup failure, not an element-type result.\n%s'
                % (
                    start_timeout,
                    solver_diagnostics(
                        run_dir,
                        job,
                    ),
                )
            )

        time.sleep(
            JOB_POLL_INTERVAL_SECONDS
        )

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

def output_paths(
    base_dir,
    element_type,
    batch_mode,
):
    base_results_dir = os.path.join(
        base_dir,
        'results',
    )

    if batch_mode:
        results_dir = os.path.join(
            base_results_dir,
            element_type,
        )
    else:
        results_dir = base_results_dir

    if not os.path.isdir(
        results_dir
    ):
        os.makedirs(
            results_dir
        )

    return {
        'results_dir': results_dir,
        'cae': os.path.join(
            results_dir,
            'connecting_lug_structure.cae',
        ),
        'odb': os.path.join(
            results_dir,
            'connecting_lug_results.odb',
        ),
        'stress': os.path.join(
            results_dir,
            'stress_contour.jpg',
        ),
        'displacement': os.path.join(
            results_dir,
            'displacement_contour.jpg',
        ),
    }

# ---------------------------------------------------------------------------
# Build, solve, post-process
# ---------------------------------------------------------------------------

def build_and_run(
    p,
    element_type=DEFAULT_ELEMENT_TYPE,
    batch_mode=False,
):
    base_dir = script_directory()

    paths = output_paths(
        base_dir,
        element_type,
        batch_mode,
    )

    cae_path = paths['cae']
    odb_path = paths['odb']
    stress_path = paths['stress']
    displacement_path = paths['displacement']
    results_dir = paths['results_dir']

    for path in (
        cae_path,
        odb_path,
        stress_path,
        displacement_path,
    ):
        if os.path.isfile(path):
            os.remove(path)

    viewport = make_viewport()

    session.journalOptions.setValues(
        replayGeometry=COORDINATE,
        recoverGeometry=COORDINATE,
    )

    original_cwd = os.getcwd()
    run_dir = None
    odb = None
    completed = False
    mdb_initialized = False

    try:
        Mdb()
        mdb_initialized = True

        mdb.models.changeKey(
            fromName='Model-1',
            toName=MODEL_NAME,
        )

        model = mdb.models[
            MODEL_NAME
        ]

        part = create_geometry(
            model,
            p,
        )

        material = model.Material(
            name='Steel',
        )

        material.Elastic(
            table=(
                (
                    YOUNGS_MODULUS,
                    POISSONS_RATIO,
                ),
            )
        )

        model.HomogeneousSolidSection(
            name='LugSection',
            material='Steel',
            thickness=1.0,
        )

        part.SectionAssignment(
            region=regionToolset.Region(
                cells=part.cells[:],
            ),
            sectionName='LugSection',
        )

        configure_mesh(
            part,
            p,
            element_type,
        )

        assembly = model.rootAssembly

        assembly.DatumCsysByDefault(
            CARTESIAN,
        )

        assembly.Instance(
            name=INSTANCE_NAME,
            part=part,
            dependent=ON,
        )

        assembly.regenerate()

        instance = assembly.instances[
            INSTANCE_NAME
        ]

        model.StaticStep(
            name=STEP_NAME,
            previous='Initial',
            description=(
                'Official lower pin-hole pressure resultant '
                'represented as nodal forces'
            ),
        )

        model.fieldOutputRequests[
            'F-Output-1'
        ].setValues(
            variables=(
                'S',
                'U',
                'RF',
                'CF',
                'NFORC',
            )
        )

        if (
            'H-Output-1'
            in model.historyOutputRequests.keys()
        ):
            del model.historyOutputRequests[
                'H-Output-1'
            ]

        fixed_face = instance.faces.findAt(
            (
                (
                    0.0,
                    0.0,
                    0.5 * p['thickness'],
                ),
            )
        )

        model.EncastreBC(
            name='FixedShank',
            createStepName='Initial',
            region=regionToolset.Region(
                faces=fixed_face,
            ),
        )

        labels = pin_load_node_labels(
            instance,
            p,
        )

        if not labels:
            raise RuntimeError(
                'No nodes were found on the lower half '
                'of the pin hole.'
            )

        load_nodes = instance.nodes.sequenceFromLabels(
            labels=tuple(labels),
        )

        assembly.Set(
            name='LOWER_PIN_HOLE_NODES',
            nodes=load_nodes,
        )

        per_node_force = (
            -p['force_magnitude']
            / float(len(labels))
        )

        model.ConcentratedForce(
            name='EquivalentPinNodalForce',
            createStepName=STEP_NAME,
            region=assembly.sets[
                'LOWER_PIN_HOLE_NODES'
            ],
            cf2=per_node_force,
        )

        viewport.setValues(
            displayedObject=assembly,
        )

        viewport.view.fitView()

        run_dir = tempfile.mkdtemp(
            prefix=(
                'abaqus_connecting_lug_%s_run_'
                % element_type.lower()
            )
        )

        cae_work_path = os.path.join(
            run_dir,
            'connecting_lug_structure.cae',
        )

        os.chdir(
            run_dir
        )

        mdb.Job(
            name=JOB_NAME,
            model=MODEL_NAME,
            type=ANALYSIS,
            description=(
                'Three-hole connecting lug with equivalent '
                'nodal pin force; element=%s'
                % element_type
            ),
            memory=90,
            memoryUnits=PERCENTAGE,
            numCpus=1,
            numDomains=1,
        )

        mdb.saveAs(
            pathName=cae_work_path,
        )

        print(
            'Submitting Abaqus job...'
        )
        print(
            'Element type: %s'
            % element_type
        )
        print(
            'Lower pin-hole load nodes: %d'
            % len(labels)
        )
        print(
            'Requested total force: Fy=%.9g N'
            % (
                -p['force_magnitude']
            )
        )

        job = mdb.jobs[
            JOB_NAME
        ]

        # Root fix:
        # Use consistency checking and DO NOT immediately call
        # waitForCompletion() while job.status may still be NONE.
        job.submit(
            consistencyChecking=ON,
        )

        wait_for_abaqus_job(
            job,
            run_dir,
        )

        if not solver_completed(
            job,
            run_dir,
        ):
            raise RuntimeError(
                'Abaqus job ended without a successful completion marker.\n%s'
                % solver_diagnostics(
                    run_dir,
                    job,
                )
            )

        odb_work_path = os.path.join(
            run_dir,
            JOB_NAME + '.odb',
        )

        if not os.path.isfile(
            odb_work_path
        ):
            raise RuntimeError(
                'The analysis completed but no ODB was found.\n%s'
                % solver_diagnostics(
                    run_dir,
                    job,
                )
            )

        odb = session.openOdb(
            name=odb_work_path,
            readOnly=True,
        )

        viewport.setValues(
            displayedObject=odb,
        )

        viewport.odbDisplay.setFrame(
            step=0,
            frame=-1,
        )

        viewport.odbDisplay.display.setValues(
            plotState=(
                CONTOURS_ON_DEF,
            ),
        )

        viewport.view.setValues(
            session.views['Iso'],
        )

        viewport.view.fitView()

        try:
            viewport.odbDisplay.commonOptions.setValues(
                visibleEdges=FEATURE,
                deformationScaling=UNIFORM,
            )

            session.printOptions.setValues(
                rendition=COLOR,
                vpDecorations=OFF,
            )

        except Exception:
            pass

        viewport.odbDisplay.setPrimaryVariable(
            variableLabel='S',
            outputPosition=INTEGRATION_POINT,
            refinement=(
                INVARIANT,
                'Mises',
            ),
        )

        print_viewport_to_jpg(
            viewport,
            os.path.join(
                run_dir,
                'stress_contour',
            ),
            stress_path,
        )

        viewport.odbDisplay.setPrimaryVariable(
            variableLabel='U',
            outputPosition=NODAL,
            refinement=(
                INVARIANT,
                'Magnitude',
            ),
        )

        print_viewport_to_jpg(
            viewport,
            os.path.join(
                run_dir,
                'displacement_contour',
            ),
            displacement_path,
        )

        odb.close()
        odb = None

        shutil.copy2(
            odb_work_path,
            odb_path,
        )

        mdb.save()

        shutil.copy2(
            cae_work_path,
            cae_path,
        )

        completed = True

    finally:
        if odb is not None:
            try:
                odb.close()
            except Exception:
                pass

        try:
            os.chdir(
                original_cwd
            )
        except Exception:
            pass

        if run_dir is not None:
            if completed:
                try:
                    shutil.rmtree(
                        run_dir
                    )
                except Exception:
                    print(
                        'Warning: temporary solver files remain in %s'
                        % run_dir
                    )
            else:
                print(
                    'Solver/mesh diagnostic files were retained in %s'
                    % run_dir
                )

        if mdb_initialized:
            try:
                mdb.close()
            except Exception:
                pass

    expected = (
        cae_path,
        odb_path,
        stress_path,
        displacement_path,
    )

    missing = [
        path
        for path in expected
        if not os.path.isfile(path)
    ]

    if missing:
        raise RuntimeError(
            'Missing requested output(s): %s'
            % ', '.join(missing)
        )

    print(
        'Analysis completed successfully for %s.'
        % element_type
    )
    print(
        'Results directory: %s'
        % results_dir
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parameters, element_types = command_line_parameters()
    batch_mode = len(element_types) > 1

    print(
        'Selected element type(s): %s'
        % ', '.join(element_types)
    )

    failures = []

    for element_type in element_types:
        print('')
        print('=' * 72)
        print(
            'Starting connecting-lug analysis with %s'
            % element_type
        )
        print('=' * 72)

        try:
            build_and_run(
                parameters,
                element_type=element_type,
                batch_mode=batch_mode,
            )

        except Exception as exc:
            failures.append(
                (
                    element_type,
                    str(exc),
                )
            )

            print(
                'FAILED for %s: %s'
                % (
                    element_type,
                    exc,
                )
            )

            if not batch_mode:
                raise

    if failures:
        lines = [
            'One or more element-type runs failed:'
        ]

        for element_type, message in failures:
            lines.append(
                '  %s: %s'
                % (
                    element_type,
                    message,
                )
            )

        raise RuntimeError(
            '\n'.join(lines)
        )

    print('')
    print(
        'All requested connecting-lug analyses completed successfully.'
    )

if __name__ == '__main__':
    main()
