# -*- coding: utf-8 -*-
"""Parametric 3-D perforated elbow-bracket analysis for Abaqus/CAE.

The solid/sketch/load workflow is derived from the official Abaqus
``flap_model.py`` (FLAP MECHANISM) example fetched with ``abaqus fetch``.
The official mechanism is intentionally reduced to one ML-friendly solid
instance while retaining curved geometry, through-holes and a true
ConcentratedForce load.  Unlike the pressure-loaded lug example, this model
distributes a prescribed total force over the mesh nodes on the free end.

SI units are used (m, Pa, N).  Supply nine comma/space separated values:

    horizontal_length, vertical_length, bend_radius, arm_width,
    hole_radius, thickness, mesh_size, force_magnitude, force_angle_deg

PowerShell example::

    $env:ELBOW_BRACKET_PARAMS = "0.12,0.10,0.05,0.04,0.008,0.012,0.008,20000,-20"
    abaqus cae noGUI=gsi_elbow_bracket_parametric_analysis.py

With no environment variable or ``--`` arguments the defaults below are
used.  Four files are written to ``results``.
"""

from __future__ import print_function

import math
import os
import shutil
import sys
import tempfile

from PIL import Image
from abaqus import *
from abaqusConstants import *
from caeModules import *
import mesh
import regionToolset

DEFAULTS = {
    'horizontal_length': 0.120,
    'vertical_length': 0.100,
    'bend_radius': 0.050,
    'arm_width': 0.040,
    'hole_radius': 0.008,
    'thickness': 0.012,
    'mesh_size': 0.008,
    'force_magnitude': 2.0E4,
    'force_angle_deg': -20.0,
}

PARAMETER_NAMES = (
    'horizontal_length',
    'vertical_length',
    'bend_radius',
    'arm_width',
    'hole_radius',
    'thickness',
    'mesh_size',
    'force_magnitude',
    'force_angle_deg',
)

MODEL_NAME = 'ElbowBracketModel'
PART_NAME = 'ElbowBracket'
INSTANCE_NAME = 'ElbowBracket-1'
STEP_NAME = 'BracketLoad'
JOB_NAME = 'ElbowBracketAnalysis'
YOUNGS_MODULUS = 200.0E9
POISSONS_RATIO = 0.30

def script_directory():
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.getcwd()

def command_line_parameters():
    raw = os.environ.get('ELBOW_BRACKET_PARAMS', '').strip()
    if raw:
        args = raw.replace(',', ' ').split()
    else:
        args = list(sys.argv[1:])
        if '--' in args:
            args = args[args.index('--') + 1:]

    if not args:
        values = dict(DEFAULTS)
    else:
        if len(args) != len(PARAMETER_NAMES):
            raise ValueError(
                'Expected nine values: %s' % ' '.join(PARAMETER_NAMES)
            )
        values = {}
        for name, text in zip(PARAMETER_NAMES, args):
            try:
                values[name] = float(text)
            except ValueError:
                raise ValueError('%s is not numeric: %s' % (name, text))

    validate_parameters(values)
    return values

def validate_parameters(p):
    positive_names = PARAMETER_NAMES[:-1]
    for name in positive_names:
        if p[name] <= 0.0:
            raise ValueError('%s must be greater than zero.' % name)

    half_width = 0.5 * p['arm_width']
    if p['bend_radius'] <= half_width:
        raise ValueError('bend_radius must be greater than arm_width/2.')
    if p['hole_radius'] >= 0.40 * p['arm_width']:
        raise ValueError('hole_radius must be smaller than 0.4*arm_width.')
    minimum_straight = 2.5 * p['arm_width']
    if p['horizontal_length'] < minimum_straight:
        raise ValueError('horizontal_length must be at least 2.5*arm_width.')
    if p['vertical_length'] < minimum_straight:
        raise ValueError('vertical_length must be at least 2.5*arm_width.')
    if p['mesh_size'] >= min(p['arm_width'], p['thickness']):
        raise ValueError('mesh_size must be smaller than arm_width and thickness.')

def make_viewport():
    name = 'Viewport: 1'
    if name not in session.viewports.keys():
        session.Viewport(name=name, origin=(0.0, 0.0), width=200.0, height=150.0)
    viewport = session.viewports[name]
    viewport.makeCurrent()
    try:
        viewport.maximize()
    except Exception:
        pass
    return viewport

def solver_completed(job, run_dir):
    if job.status == COMPLETED:
        return True
    sta_path = os.path.join(run_dir, JOB_NAME + '.sta')
    if not os.path.isfile(sta_path):
        return False
    stream = open(sta_path, 'rb')
    try:
        return b'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in stream.read().upper()
    finally:
        stream.close()

def print_viewport_to_jpg(viewport, temporary_base, jpg_path):
    png_path = temporary_base + '.png'
    session.printToFile(
        fileName=temporary_base,
        format=PNG,
        canvasObjects=(viewport,)
    )
    rendered = Image.open(png_path)
    try:
        rendered.convert('RGB').save(jpg_path, format='JPEG', quality=95)
    finally:
        rendered.close()
    os.remove(png_path)

def create_geometry(model, p):
    h = p['horizontal_length']
    v = p['vertical_length']
    r = p['bend_radius']
    w2 = 0.5 * p['arm_width']
    end_y = r + v

    sheet_size = 4.0 * max(h + r + w2, end_y, p['thickness'])
    sketch = model.ConstrainedSketch(name='__profile__', sheetSize=sheet_size)

    # Constant-width 90-degree elbow: two straight legs joined by concentric
    # quarter-circle arcs.  Two through-holes make the topology non-trivial.
    sketch.Line(point1=(0.0, -w2), point2=(h, -w2))
    sketch.ArcByCenterEnds(
        center=(h, r),
        point1=(h, -w2),
        point2=(h + r + w2, r),
        direction=COUNTERCLOCKWISE
    )
    sketch.Line(point1=(h + r + w2, r), point2=(h + r + w2, end_y))
    sketch.Line(point1=(h + r + w2, end_y), point2=(h + r - w2, end_y))
    sketch.Line(point1=(h + r - w2, end_y), point2=(h + r - w2, r))
    sketch.ArcByCenterEnds(
        center=(h, r),
        point1=(h + r - w2, r),
        point2=(h, w2),
        direction=CLOCKWISE
    )
    sketch.Line(point1=(h, w2), point2=(0.0, w2))
    sketch.Line(point1=(0.0, w2), point2=(0.0, -w2))

    hole_r = p['hole_radius']
    hole1 = (0.38 * h, 0.0)
    hole2 = (h + r, r + 0.62 * v)
    sketch.CircleByCenterPerimeter(
        center=hole1,
        point1=(hole1[0] + hole_r, hole1[1])
    )
    sketch.CircleByCenterPerimeter(
        center=hole2,
        point1=(hole2[0] + hole_r, hole2[1])
    )

    part = model.Part(
        name=PART_NAME,
        dimensionality=THREE_D,
        type=DEFORMABLE_BODY
    )
    part.BaseSolidExtrude(sketch=sketch, depth=p['thickness'])
    del model.sketches['__profile__']
    return part, end_y

def build_and_run(p):
    base_dir = script_directory()
    results_dir = os.path.join(base_dir, 'results')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    cae_path = os.path.join(results_dir, 'elbow_bracket_structure.cae')
    odb_path = os.path.join(results_dir, 'elbow_bracket_results.odb')
    stress_path = os.path.join(results_dir, 'stress_contour.jpg')
    displacement_path = os.path.join(results_dir, 'displacement_contour.jpg')
    for path in (cae_path, odb_path, stress_path, displacement_path):
        if os.path.isfile(path):
            os.remove(path)

    viewport = make_viewport()
    session.journalOptions.setValues(
        replayGeometry=COORDINATE,
        recoverGeometry=COORDINATE
    )
    Mdb()
    mdb.models.changeKey(fromName='Model-1', toName=MODEL_NAME)
    model = mdb.models[MODEL_NAME]

    part, end_y = create_geometry(model, p)
    viewport.setValues(displayedObject=part)
    viewport.view.fitView()

    material = model.Material(name='Steel')
    material.Elastic(table=((YOUNGS_MODULUS, POISSONS_RATIO),))
    model.HomogeneousSolidSection(
        name='BracketSection', material='Steel', thickness=1.0
    )
    part.SectionAssignment(
        region=regionToolset.Region(cells=part.cells[:]),
        sectionName='BracketSection'
    )

    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(name=INSTANCE_NAME, part=part, dependent=ON)
    assembly.regenerate()
    instance = assembly.instances[INSTANCE_NAME]

    model.StaticStep(
        name=STEP_NAME,
        previous='Initial',
        description='Concentrated nodal forces on the free end of a perforated elbow bracket'
    )
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U', 'RF', 'CF', 'NFORC')
    )
    if 'H-Output-1' in model.historyOutputRequests.keys():
        del model.historyOutputRequests['H-Output-1']

    fixed_face = instance.faces.findAt(((0.0, 0.0, 0.5 * p['thickness']),))
    model.EncastreBC(
        name='FixedEnd',
        createStepName=STEP_NAME,
        region=regionToolset.Region(faces=fixed_face)
    )

    part.seedPart(
        size=p['mesh_size'],
        deviationFactor=0.1,
        minSizeFactor=0.1
    )
    part.setMeshControls(regions=part.cells[:], elemShape=TET, technique=FREE)
    elem_type = mesh.ElemType(elemCode=C3D10M, elemLibrary=STANDARD)
    part.setElementType(regions=(part.cells[:],), elemTypes=(elem_type,))
    part.generateMesh()
    assembly.regenerate()

    # Select all mesh nodes on the top end.  Abaqus applies one concentrated
    # force vector to every node in the set, so divide the requested total by
    # the node count.  The exported node-force rows therefore sum exactly to
    # force_magnitude (apart from floating-point rounding).
    instance = assembly.instances[INSTANCE_NAME]
    tol = max(1.0E-9, p['mesh_size'] * 1.0E-3)
    load_nodes = instance.nodes.getByBoundingBox(
        xMin=p['horizontal_length'] + p['bend_radius'] - 0.5 * p['arm_width'] - tol,
        xMax=p['horizontal_length'] + p['bend_radius'] + 0.5 * p['arm_width'] + tol,
        yMin=end_y - tol,
        yMax=end_y + tol,
        zMin=-tol,
        zMax=p['thickness'] + tol
    )
    if len(load_nodes) == 0:
        raise RuntimeError('No mesh nodes were found on the free end.')

    assembly.Set(name='LOAD_NODES', nodes=load_nodes)
    angle = math.radians(p['force_angle_deg'])
    total_fx = p['force_magnitude'] * math.cos(angle)
    total_fy = p['force_magnitude'] * math.sin(angle)
    per_node_fx = total_fx / float(len(load_nodes))
    per_node_fy = total_fy / float(len(load_nodes))
    model.ConcentratedForce(
        name='EndNodalForce',
        createStepName=STEP_NAME,
        region=assembly.sets['LOAD_NODES'],
        cf1=per_node_fx,
        cf2=per_node_fy
    )

    viewport.setValues(displayedObject=assembly)
    viewport.view.fitView()

    run_dir = tempfile.mkdtemp(prefix='abaqus_elbow_bracket_run_')
    cae_work_path = os.path.join(run_dir, 'elbow_bracket_structure.cae')
    original_cwd = os.getcwd()
    odb = None
    completed = False

    try:
        os.chdir(run_dir)
        mdb.Job(
            name=JOB_NAME,
            model=MODEL_NAME,
            type=ANALYSIS,
            description='Linear elastic perforated elbow bracket with nodal end forces',
            memory=90,
            memoryUnits=PERCENTAGE,
            numCpus=1,
            numDomains=1
        )
        mdb.saveAs(pathName=cae_work_path)

        print('Submitting Abaqus job...')
        print('Load nodes: %d' % len(load_nodes))
        print('Requested total force: Fx=%.9g N, Fy=%.9g N' % (total_fx, total_fy))
        job = mdb.jobs[JOB_NAME]
        job.submit(consistencyChecking=OFF)
        job.waitForCompletion()
        if not solver_completed(job, run_dir):
            raise RuntimeError('Abaqus job ended with status: %s' % job.status)

        odb_work_path = os.path.join(run_dir, JOB_NAME + '.odb')
        if not os.path.isfile(odb_work_path):
            raise RuntimeError('The analysis completed but no ODB was found.')

        odb = session.openOdb(name=odb_work_path, readOnly=True)
        viewport.setValues(displayedObject=odb)
        viewport.odbDisplay.setFrame(step=0, frame=-1)
        viewport.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF,))
        viewport.view.setValues(session.views['Iso'])
        viewport.view.fitView()
        try:
            viewport.odbDisplay.commonOptions.setValues(
                visibleEdges=FEATURE, deformationScaling=UNIFORM
            )
            session.printOptions.setValues(rendition=COLOR, vpDecorations=OFF)
        except Exception:
            pass

        viewport.odbDisplay.setPrimaryVariable(
            variableLabel='S',
            outputPosition=INTEGRATION_POINT,
            refinement=(INVARIANT, 'Mises')
        )
        print_viewport_to_jpg(
            viewport, os.path.join(run_dir, 'stress_contour'), stress_path
        )
        viewport.odbDisplay.setPrimaryVariable(
            variableLabel='U',
            outputPosition=NODAL,
            refinement=(INVARIANT, 'Magnitude')
        )
        print_viewport_to_jpg(
            viewport, os.path.join(run_dir, 'displacement_contour'), displacement_path
        )
        odb.close()
        odb = None

        shutil.copy2(odb_work_path, odb_path)
        mdb.save()
        shutil.copy2(cae_work_path, cae_path)
        completed = True
    finally:
        if odb is not None:
            try:
                odb.close()
            except Exception:
                pass
        os.chdir(original_cwd)
        if completed:
            try:
                mdb.close()
            except Exception:
                pass
            try:
                shutil.rmtree(run_dir)
            except Exception:
                print('Warning: temporary solver files remain in %s' % run_dir)
        else:
            print('Solver diagnostic files were retained in %s' % run_dir)

    expected = (cae_path, odb_path, stress_path, displacement_path)
    missing = [path for path in expected if not os.path.isfile(path)]
    if missing:
        raise RuntimeError('Missing requested output(s): %s' % ', '.join(missing))

    print('Analysis completed successfully.')
    print('Results directory: %s' % results_dir)
    for path in expected:
        print('  %s' % os.path.basename(path))

if __name__ == '__main__':
    build_and_run(command_line_parameters())
