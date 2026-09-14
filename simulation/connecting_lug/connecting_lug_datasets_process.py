# -*- coding: utf-8 -*-
"""Convert connecting_lug_i CAE/ODB samples to machine-learning CSV files.

This is a standalone replacement for the previous front end that imported
``lug_datasets_process.py``.  It contains all required extraction/writing
routines in this file.

Run with the Abaqus/CAE Python kernel:
abaqus cae noGUI=connecting_lug_datasets_process.py

Optional arguments after ``--`` include ``--input-dir``, ``--output-dir``,
``--instance``, ``--model``, ``--step``, ``--frame``,
``--vertex-tolerance``, ``--force-zero-tolerance`` and ``--fail-fast``.

Expected sample layout::

    connecting_lug/
        connecting_lug_1/
            connecting_lug_structure.cae
            connecting_lug_results.odb
        connecting_lug_2/
            ...

Output layout::

    connecting_lug_processed/
        input_coord/coord_1.csv
        input_matrix/matrix_1.csv
        input_force/force_1.csv
        output_displace/displace_1.csv
        output_stress/stress_1.csv
        ...

Notes
-----
* ConcentratedForce values are read from the active CAE step state
  (``step.loadStates``), so propagated/modified values are respected.
* A concentrated force applied to several nodes/vertices is written to every
  mapped node, matching Abaqus CLOAD semantics.
* CAE mesh nodes are mapped by instance name + node label whenever possible.
  Geometric vertices are first mapped through ``Vertex.getNodes()``; if that
  is unavailable, their coordinates are mapped to the nearest ODB node.
* Stress ``S`` is requested at ``ELEMENT_NODAL`` position and arithmetic-
  averaged over all element-nodal contributions that share a node.
* This script assumes force components are already defined in the global
  coordinate system.  A warning is printed if a ConcentratedForce uses a
  local coordinate system.
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import re
import sys
import traceback

from abaqus import mdb, openMdb
from abaqusConstants import ELEMENT_NODAL, NODAL
from odbAccess import openOdb

SAMPLE_PATTERN = re.compile(r'^connecting_lug_(\d+)$', re.IGNORECASE)

DATASET_SUBDIRS = {
    'coord': 'input_coord',
    'matrix': 'input_matrix',
    'force': 'input_force',
    'displace': 'output_displace',
    'stress': 'output_stress',
}

try:
    string_types = (basestring,)
except NameError:
    string_types = (str, bytes)

INACTIVE_LOAD_STATUSES = set((
    'NOT_YET_ACTIVE',
    'DEACTIVATED',
    'NO_LONGER_ACTIVE',
    'TYPE_NOT_APPLICABLE',
    'INSTANCE_NOT_APPLICABLE',
))

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def log(message):
    sys.stdout.write(str(message) + '\n')
    sys.stdout.flush()

def ensure_dir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise

def _repo_keys(repo):
    try:
        return list(repo.keys())
    except Exception:
        return []

def _repo_get_case_insensitive(repo, requested_name):
    """Return (actual_key, value) from an Abaqus repository."""
    if requested_name is None:
        return None, None

    try:
        if requested_name in repo:
            return requested_name, repo[requested_name]
    except Exception:
        pass

    target = str(requested_name).lower()
    for key in _repo_keys(repo):
        if str(key).lower() == target:
            return key, repo[key]
    return None, None

def _same_name(a, b):
    if a is None or b is None:
        return False
    return str(a).lower() == str(b).lower()

def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return 0

def _to_real_float(value, description):
    """Convert Abaqus numeric values to a real float with a useful error."""
    try:
        z = complex(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            raise TypeError('Could not convert %s=%r to float.' % (description, value))

    if abs(z.imag) > 1.0e-12 * max(1.0, abs(z.real)):
        raise ValueError(
            '%s is complex (%r). This exporter expects real-valued static loads/results.'
            % (description, value)
        )
    return float(z.real)

def _vector3(values):
    data = list(values)
    while len(data) < 3:
        data.append(0.0)
    return (
        float(data[0]),
        float(data[1]),
        float(data[2]),
    )

def _field_value_data(value):
    """Read FieldValue data in both single- and double-precision ODBs."""
    for attr in ('data', 'dataDouble'):
        try:
            data = getattr(value, attr)
        except Exception:
            continue
        if data is not None:
            try:
                return tuple(data)
            except TypeError:
                return (data,)
    raise RuntimeError('Could not read field value data from %r.' % (value,))

def _value_belongs_to_instance(value, odb_instance):
    try:
        value_instance = value.instance
    except Exception:
        value_instance = None
    if value_instance is None:
        return True
    return _same_name(getattr(value_instance, 'name', None), odb_instance.name)

# ---------------------------------------------------------------------------
# tqdm compatibility (Abaqus installations do not always ship tqdm)
# ---------------------------------------------------------------------------

try:
    from tqdm import tqdm as _real_tqdm
except Exception:
    _real_tqdm = None

class _FallbackProgress(object):
    def __init__(self, iterable, total=None, desc=None, unit=None, **_kwargs):
        self.iterable = iterable
        self.total = total if total is not None else _safe_len(iterable)
        self.desc = desc or 'Processing'
        self.unit = unit or 'item'
        self.index = 0
        self.postfix = ''

    def __iter__(self):
        for item in self.iterable:
            self.index += 1
            if self.total:
                log('%s: %d/%d %s%s' % (
                    self.desc,
                    self.index,
                    self.total,
                    self.unit,
                    self.postfix,
                ))
            else:
                log('%s: %d %s%s' % (
                    self.desc,
                    self.index,
                    self.unit,
                    self.postfix,
                ))
            yield item

    def set_postfix(self, **kwargs):
        if kwargs:
            pieces = []
            for key in sorted(kwargs):
                pieces.append('%s=%s' % (key, kwargs[key]))
            self.postfix = ' [' + ', '.join(pieces) + ']'

def tqdm(iterable, **kwargs):
    if _real_tqdm is not None:
        return _real_tqdm(iterable, **kwargs)
    return _FallbackProgress(iterable, **kwargs)

# ---------------------------------------------------------------------------
# ODB selection / extraction
# ---------------------------------------------------------------------------

def open_odb_read_only(path):
    return openOdb(path=path, readOnly=True)

def choose_odb_instance(odb, requested_instance=None):
    instances = odb.rootAssembly.instances

    if requested_instance:
        actual_name, instance = _repo_get_case_insensitive(
            instances, requested_instance
        )
        if instance is None:
            raise KeyError(
                'ODB instance %r not found. Available instances: %s'
                % (requested_instance, ', '.join(map(str, _repo_keys(instances))))
            )
        return instance

    candidates = []
    for name in _repo_keys(instances):
        instance = instances[name]
        node_count = _safe_len(getattr(instance, 'nodes', ()))
        elem_count = _safe_len(getattr(instance, 'elements', ()))
        if node_count > 0 and elem_count > 0:
            candidates.append((elem_count, node_count, str(name), instance))

    if not candidates:
        raise RuntimeError('No ODB instance containing both nodes and elements was found.')

    if len(candidates) == 1:
        return candidates[0][3]

    # A bracket CAE can occasionally contain auxiliary/rigid instances.  The
    # structural instance is normally the one with the largest element mesh.
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    chosen = candidates[-1][3]
    log(
        '[warning] Multiple meshed ODB instances found; automatically using '
        '%r (largest element count). Use --instance to override.' % chosen.name
    )
    return chosen

def choose_odb_step_and_frame(odb, requested_step=None, frame_index=-1):
    steps = odb.steps
    step_names = _repo_keys(steps)
    if not step_names:
        raise RuntimeError('ODB contains no analysis steps.')

    if requested_step:
        actual_name, step = _repo_get_case_insensitive(steps, requested_step)
        if step is None:
            raise KeyError(
                'ODB step %r not found. Available steps: %s'
                % (requested_step, ', '.join(map(str, step_names)))
            )
    else:
        actual_name = step_names[-1]
        step = steps[actual_name]

    frames = step.frames
    nframes = len(frames)
    if nframes == 0:
        raise RuntimeError('ODB step %r contains no frames.' % actual_name)

    index = int(frame_index)
    if index < 0:
        index += nframes
    if index < 0 or index >= nframes:
        raise IndexError(
            'Frame index %d is out of range for step %r, which has %d frames.'
            % (frame_index, actual_name, nframes)
        )

    return actual_name, frames[index]

def extract_mesh(odb_instance):
    nodes = sorted(list(odb_instance.nodes), key=lambda obj: int(obj.label))
    elements = sorted(list(odb_instance.elements), key=lambda obj: int(obj.label))

    if not nodes:
        raise RuntimeError('Selected ODB instance %r has no nodes.' % odb_instance.name)
    if not elements:
        raise RuntimeError('Selected ODB instance %r has no elements.' % odb_instance.name)

    node_labels = []
    coord_by_label = {}
    coord_rows = []

    for node in nodes:
        label = int(node.label)
        xyz = _vector3(node.coordinates)
        node_labels.append(label)
        coord_by_label[label] = xyz
        coord_rows.append((label, xyz[0], xyz[1], xyz[2]))

    max_connectivity = 0
    connectivity_rows = []
    for element in elements:
        conn = tuple(int(label) for label in element.connectivity)
        if len(conn) > max_connectivity:
            max_connectivity = len(conn)
        connectivity_rows.append((int(element.label), conn))

    if max_connectivity == 0:
        raise RuntimeError('Selected ODB instance contains elements with empty connectivity.')

    matrix_rows = []
    for element_label, conn in connectivity_rows:
        padded = list(conn) + [''] * (max_connectivity - len(conn))
        matrix_rows.append(tuple([element_label] + padded))

    coord_header = ('node', 'x', 'y', 'z')
    matrix_header = tuple(
        ['element'] + ['node%d' % (i + 1) for i in range(max_connectivity)]
    )

    return (
        node_labels,
        coord_by_label,
        coord_header,
        coord_rows,
        matrix_header,
        matrix_rows,
    )

def extract_displacement(frame, odb_instance, node_labels):
    if 'U' not in frame.fieldOutputs:
        raise KeyError(
            'Field output U (displacement) is not present in the selected ODB frame.'
        )

    field = frame.fieldOutputs['U']
    try:
        subset = field.getSubset(region=odb_instance)
    except Exception:
        subset = field

    by_label = {}
    for value in subset.values:
        if not _value_belongs_to_instance(value, odb_instance):
            continue
        try:
            label = int(value.nodeLabel)
        except Exception:
            continue
        data = _vector3(_field_value_data(value))
        by_label[label] = data

    missing = [label for label in node_labels if label not in by_label]
    if missing:
        preview = ', '.join(map(str, missing[:10]))
        raise RuntimeError(
            'Displacement U is missing for %d nodes in instance %r. First missing labels: %s'
            % (len(missing), odb_instance.name, preview)
        )

    rows = []
    for label in node_labels:
        u = by_label[label]
        rows.append((label, u[0], u[1], u[2]))

    return ('node', 'dx', 'dy', 'dz'), rows

def _stress_element_nodal_subset(field, odb_instance):
    try:
        return field.getSubset(region=odb_instance, position=ELEMENT_NODAL)
    except Exception:
        # Some Abaqus releases are happier with chained getSubset calls.
        regional = field.getSubset(region=odb_instance)
        return regional.getSubset(position=ELEMENT_NODAL)

def _stress_nodal_subset(field, odb_instance):
    try:
        return field.getSubset(region=odb_instance, position=NODAL)
    except Exception:
        regional = field.getSubset(region=odb_instance)
        return regional.getSubset(position=NODAL)

def extract_stress(frame, odb_instance, node_labels):
    if 'S' not in frame.fieldOutputs:
        raise KeyError(
            'Field output S (stress) is not present in the selected ODB frame.'
        )

    field = frame.fieldOutputs['S']

    try:
        subset = _stress_element_nodal_subset(field, odb_instance)
        values = list(subset.values)
    except Exception:
        subset = None
        values = []

    # If ELEMENT_NODAL could not be produced, use explicitly stored NODAL S.
    if not values:
        try:
            subset = _stress_nodal_subset(field, odb_instance)
            values = list(subset.values)
        except Exception:
            values = []

    if not values:
        raise RuntimeError(
            'Could not obtain stress S at ELEMENT_NODAL or NODAL position for instance %r.'
            % odb_instance.name
        )

    component_labels = list(getattr(subset, 'componentLabels', ()) or ())
    if not component_labels:
        component_labels = list(getattr(field, 'componentLabels', ()) or ())

    sums = {}
    counts = {}
    component_count = None

    for value in values:
        if not _value_belongs_to_instance(value, odb_instance):
            continue
        try:
            label = int(value.nodeLabel)
        except Exception:
            continue

        data = tuple(float(x) for x in _field_value_data(value))
        if component_count is None:
            component_count = len(data)
        elif len(data) != component_count:
            raise RuntimeError(
                'Stress component count changed within the selected field: %d vs %d.'
                % (component_count, len(data))
            )

        if label not in sums:
            sums[label] = [0.0] * len(data)
            counts[label] = 0
        for i, number in enumerate(data):
            sums[label][i] += number
        counts[label] += 1

    if component_count is None:
        raise RuntimeError('No nodal stress values were found for the selected instance.')

    if len(component_labels) != component_count:
        raise RuntimeError(
            'Stress component labels are unavailable or inconsistent: got %d labels '
            'for %d components.' % (len(component_labels), component_count)
        )

    component_index = {}
    for index, name in enumerate(component_labels):
        component_index[str(name).upper()] = index

    required_components = ('S11', 'S22', 'S33', 'S12', 'S23', 'S13')
    missing_components = [
        name for name in required_components if name not in component_index
    ]
    if missing_components:
        raise RuntimeError(
            'The requested 3D stress CSV requires %s; missing %s. Available '
            'components: %s'
            % (
                ', '.join(required_components),
                ', '.join(missing_components),
                ', '.join(map(str, component_labels)),
            )
        )

    missing = [label for label in node_labels if label not in sums]
    if missing:
        preview = ', '.join(map(str, missing[:10]))
        raise RuntimeError(
            'Stress S is missing for %d nodes in instance %r. First missing labels: %s'
            % (len(missing), odb_instance.name, preview)
        )

    rows = []
    for label in node_labels:
        count = float(counts[label])
        averaged = [number / count for number in sums[label]]
        xx = averaged[component_index['S11']]
        yy = averaged[component_index['S22']]
        zz = averaged[component_index['S33']]
        xy = averaged[component_index['S12']]
        yz = averaged[component_index['S23']]
        zx = averaged[component_index['S13']]
        von_mises = math.sqrt(
            0.5 * (
                (xx - yy) * (xx - yy)
                + (yy - zz) * (yy - zz)
                + (zz - xx) * (zz - xx)
            )
            + 3.0 * (xy * xy + yz * yz + zx * zx)
        )
        rows.append((label, xx, yy, zz, xy, yz, zx, von_mises))

    return ('node', 'xx', 'yy', 'zz', 'xy', 'yz', 'zx', 'von_mises'), rows

# ---------------------------------------------------------------------------
# CAE model / load selection
# ---------------------------------------------------------------------------

def choose_model(opened_mdb, odb_path, requested_model=None):
    models = opened_mdb.models

    if requested_model:
        actual_name, model = _repo_get_case_insensitive(models, requested_model)
        if model is None:
            raise KeyError(
                'CAE model %r not found. Available models: %s'
                % (requested_model, ', '.join(map(str, _repo_keys(models))))
            )
        return model

    model_names = _repo_keys(models)
    if len(model_names) == 1:
        return models[model_names[0]]

    # If the ODB basename matches a saved CAE job, the job tells us the model.
    odb_stem = os.path.splitext(os.path.basename(odb_path))[0]
    jobs = getattr(opened_mdb, 'jobs', None)
    if jobs is not None:
        actual_job_name, job = _repo_get_case_insensitive(jobs, odb_stem)
        if job is not None:
            job_model = getattr(job, 'model', None)
            if hasattr(job_model, 'name'):
                job_model = job_model.name
            if job_model:
                actual_model_name, model = _repo_get_case_insensitive(
                    models, str(job_model)
                )
                if model is not None:
                    return model

    raise RuntimeError(
        'CAE file contains multiple models (%s), and the ODB filename did not '
        'identify a matching CAE job. Please pass --model.'
        % ', '.join(map(str, model_names))
    )

def choose_cae_step(model, odb_step_name):
    steps = model.steps
    actual_name, step = _repo_get_case_insensitive(steps, odb_step_name)
    if step is not None:
        return actual_name, step

    names = [name for name in _repo_keys(steps) if str(name).lower() != 'initial']
    if len(names) == 1:
        log(
            '[warning] ODB step %r was not found by name in the CAE model; '
            'using the only non-Initial CAE step %r.' % (odb_step_name, names[0])
        )
        return names[0], steps[names[0]]

    raise KeyError(
        'Could not match ODB step %r to a CAE step. Available CAE steps: %s'
        % (odb_step_name, ', '.join(map(str, _repo_keys(steps))))
    )

def _collect_repository_owners(value, output, visited):
    """Collect objects from a Region tuple that can own sets."""
    obj_id = id(value)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if value is None:
        return

    if any(hasattr(value, attr) for attr in ('sets', 'allSets', 'allInternalSets')):
        output.append(value)
        return

    if isinstance(value, string_types):
        return

    try:
        iterator = iter(value)
    except Exception:
        return

    for item in iterator:
        _collect_repository_owners(item, output, visited)

def _find_set_on_owner(owner, set_name):
    for attr in ('sets', 'allSets', 'allInternalSets'):
        repo = getattr(owner, attr, None)
        if repo is None:
            continue
        actual_name, region_set = _repo_get_case_insensitive(repo, set_name)
        if region_set is not None:
            return region_set
    return None

def _resolve_load_region_set(model, load, selected_instance_name):
    """Resolve load.region (a Region tuple) back to its Set object."""
    region = getattr(load, 'region', None)
    if region is None:
        raise RuntimeError('Load %r has no region.' % getattr(load, 'name', '<unnamed>'))

    # Some APIs/test doubles expose a Set-like object directly.
    if any(hasattr(region, attr) for attr in ('nodes', 'vertices', 'referencePoints')):
        return region

    try:
        set_name = region[0]
    except Exception:
        raise RuntimeError(
            'Could not interpret region for load %r: %r'
            % (getattr(load, 'name', '<unnamed>'), region)
        )

    owners = []
    _collect_repository_owners(region[1:], owners, set())

    assembly = model.rootAssembly

    # Prefer owners encoded in the Region tuple itself.
    for owner in owners:
        region_set = _find_set_on_owner(owner, set_name)
        if region_set is not None:
            return region_set

    # Then prefer the selected structural instance.
    actual_name, selected_instance = _repo_get_case_insensitive(
        assembly.instances, selected_instance_name
    )
    if selected_instance is not None:
        region_set = _find_set_on_owner(selected_instance, set_name)
        if region_set is not None:
            return region_set

    # Assembly-level sets include the common case of picked/internal load sets.
    region_set = _find_set_on_owner(assembly, set_name)
    if region_set is not None:
        return region_set

    # Last-resort search through all instances and parts.
    for name in _repo_keys(assembly.instances):
        region_set = _find_set_on_owner(assembly.instances[name], set_name)
        if region_set is not None:
            return region_set

    parts = getattr(model, 'parts', None)
    if parts is not None:
        for name in _repo_keys(parts):
            region_set = _find_set_on_owner(parts[name], set_name)
            if region_set is not None:
                return region_set

    raise RuntimeError(
        'Could not resolve region set %r for load %r.'
        % (set_name, getattr(load, 'name', '<unnamed>'))
    )

def _iter_entities(container):
    """Flatten Abaqus entity arrays/tuples without assuming their exact type."""
    if container is None:
        return

    # Entity objects themselves are leaves.
    if (
        hasattr(container, 'label')
        or hasattr(container, 'pointOn')
        or hasattr(container, 'point')
    ):
        yield container
        return

    if isinstance(container, string_types):
        return

    try:
        iterator = iter(container)
    except Exception:
        return

    for item in iterator:
        for entity in _iter_entities(item):
            yield entity

def _entity_instance_name(entity):
    try:
        name = entity.instanceName
    except Exception:
        return None
    if name is None:
        return None
    try:
        return str(name)
    except Exception:
        return name

def _entity_coordinates(entity, assembly=None):
    if assembly is not None:
        try:
            return _vector3(assembly.getCoordinates(entity=entity))
        except Exception:
            pass

    for attr in ('coordinates', 'pointOn', 'point'):
        try:
            value = getattr(entity, attr)
        except Exception:
            continue
        if value is None:
            continue

        # Vertex.pointOn is (x, y, z); some geometry objects use ((x,y,z), ...).
        try:
            if len(value) > 0 and hasattr(value[0], '__iter__'):
                value = value[0]
        except Exception:
            pass

        try:
            return _vector3(value)
        except Exception:
            continue

    return None

def _default_vertex_tolerance(coord_by_label):
    coords = list(coord_by_label.values())
    if not coords:
        return 1.0e-8

    mins = [min(point[i] for point in coords) for i in range(3)]
    maxs = [max(point[i] for point in coords) for i in range(3)]
    # Abaqus/CAE executes noGUI scripts in ``__main__.__dict__``.  That
    # namespace can contain an Abaqus ``sum`` command which shadows Python's
    # built-in sum() and rejects generator arguments.  Spell out this fixed
    # three-dimensional norm so it behaves identically in plain Python and in
    # the CAE kernel.
    dx = maxs[0] - mins[0]
    dy = maxs[1] - mins[1]
    dz = maxs[2] - mins[2]
    diagonal = math.sqrt(dx * dx + dy * dy + dz * dz)

    # ODB coordinates can be single precision while CAE geometry is double
    # precision.  A relative 1e-6 tolerance is small compared with ordinary
    # FE element sizes but large enough for representation round-off.
    return max(1.0e-8, diagonal * 1.0e-6)

def _nearest_odb_label(point, coord_items, tolerance):
    best_label = None
    best_dist2 = None

    for label, xyz in coord_items:
        dx = point[0] - xyz[0]
        dy = point[1] - xyz[1]
        dz = point[2] - xyz[2]
        dist2 = dx * dx + dy * dy + dz * dz
        if best_dist2 is None or dist2 < best_dist2 or (
            dist2 == best_dist2 and int(label) < int(best_label)
        ):
            best_dist2 = dist2
            best_label = label

    if best_label is None:
        return None

    if best_dist2 > tolerance * tolerance:
        return None
    return int(best_label)

def _map_mesh_node_to_odb(
    node,
    selected_instance_name,
    coord_by_label,
    coord_items,
    tolerance,
    assembly,
):
    instance_name = _entity_instance_name(node)
    if instance_name and not _same_name(instance_name, selected_instance_name):
        return None

    try:
        label = int(node.label)
    except Exception:
        label = None

    if label is not None and label in coord_by_label:
        return label

    point = _entity_coordinates(node, assembly=assembly)
    if point is None:
        return None
    return _nearest_odb_label(point, coord_items, tolerance)

def _map_vertex_to_odb(
    vertex,
    selected_instance_name,
    coord_by_label,
    coord_items,
    tolerance,
    assembly,
):
    instance_name = _entity_instance_name(vertex)
    if instance_name and not _same_name(instance_name, selected_instance_name):
        return []

    mapped = []

    # Best path: ask CAE which mesh nodes are attached to this geometric vertex.
    try:
        attached_nodes = vertex.getNodes()
    except Exception:
        attached_nodes = ()

    for node in _iter_entities(attached_nodes):
        label = _map_mesh_node_to_odb(
            node,
            selected_instance_name,
            coord_by_label,
            coord_items,
            tolerance,
            assembly,
        )
        if label is not None:
            mapped.append(label)

    if mapped:
        return sorted(set(mapped))

    # Fallback for geometry-only regions or unavailable vertex->mesh links.
    point = _entity_coordinates(vertex, assembly=assembly)
    if point is None:
        return []
    label = _nearest_odb_label(point, coord_items, tolerance)
    if label is None:
        return []
    return [label]

def _active_concentrated_force_states(model, cae_step):
    states = getattr(cae_step, 'loadStates', None)
    if states is None:
        return

    loads = getattr(model, 'loads', None)
    if loads is None:
        return

    for load_name in _repo_keys(loads):
        load = loads[load_name]

        # Skip suppressed load definitions.
        try:
            if bool(load.suppressed):
                continue
        except Exception:
            pass

        actual_state_name, state = _repo_get_case_insensitive(states, load_name)
        if state is None:
            continue

        # ConcentratedForceState is identified by its cf1/cf2/cf3 members.
        if not all(hasattr(state, attr) for attr in ('cf1', 'cf2', 'cf3')):
            continue
        if not hasattr(load, 'region'):
            continue

        status = str(getattr(state, 'status', '')).upper()
        if status in INACTIVE_LOAD_STATUSES:
            continue

        yield load_name, load, state

def extract_forces(
    model,
    cae_step,
    odb,
    odb_instance,
    coord_by_label,
    vertex_tolerance=None,
    zero_tolerance=0.0,
):
    del odb  # Kept in the signature for compatibility with the previous core API.

    if zero_tolerance < 0.0:
        raise ValueError('--force-zero-tolerance must be non-negative.')

    if vertex_tolerance is None:
        tolerance = _default_vertex_tolerance(coord_by_label)
    else:
        tolerance = float(vertex_tolerance)
        if tolerance < 0.0:
            raise ValueError('--vertex-tolerance must be non-negative.')

    coord_items = sorted(coord_by_label.items())
    assembly = model.rootAssembly
    selected_instance_name = str(odb_instance.name)
    force_by_label = {}

    for load_name, load, state in _active_concentrated_force_states(model, cae_step):
        distribution = str(getattr(load, 'distributionType', 'UNIFORM')).upper()
        if distribution == 'FIELD':
            raise NotImplementedError(
                'ConcentratedForce %r uses FIELD distribution. This standalone '
                'exporter intentionally does not guess analytical-field scaling.'
                % load_name
            )

        local_csys = getattr(load, 'localCsys', None)
        if local_csys is not None and str(local_csys).upper() not in ('NONE', ''):
            log(
                '[warning] ConcentratedForce %r uses localCsys=%r. cf1/cf2/cf3 '
                'are exported as stored; verify that this matches your ML '
                'coordinate convention.' % (load_name, local_csys)
            )

        force = (
            _to_real_float(state.cf1, '%s.cf1' % load_name),
            _to_real_float(state.cf2, '%s.cf2' % load_name),
            _to_real_float(state.cf3, '%s.cf3' % load_name),
        )

        # Completely zero active loads do not contribute anything.
        if max(abs(force[0]), abs(force[1]), abs(force[2])) <= zero_tolerance:
            continue

        region_set = _resolve_load_region_set(
            model, load, selected_instance_name
        )

        target_labels = set()

        for node in _iter_entities(getattr(region_set, 'nodes', None)):
            label = _map_mesh_node_to_odb(
                node,
                selected_instance_name,
                coord_by_label,
                coord_items,
                tolerance,
                assembly,
            )
            if label is not None:
                target_labels.add(label)

        for vertex in _iter_entities(getattr(region_set, 'vertices', None)):
            labels = _map_vertex_to_odb(
                vertex,
                selected_instance_name,
                coord_by_label,
                coord_items,
                tolerance,
                assembly,
            )
            target_labels.update(labels)

        if not target_labels:
            log(
                '[warning] Active ConcentratedForce %r did not map to any node '
                'of ODB instance %r and was skipped. If it is applied to a '
                'reference point or another instance, this can be expected.'
                % (load_name, selected_instance_name)
            )
            continue

        for label in target_labels:
            previous = force_by_label.get(label, (0.0, 0.0, 0.0))
            force_by_label[label] = (
                previous[0] + force[0],
                previous[1] + force[1],
                previous[2] + force[2],
            )

    rows = []
    for label in sorted(force_by_label):
        force = list(force_by_label[label])
        for i in range(3):
            if abs(force[i]) <= zero_tolerance:
                force[i] = 0.0
        if max(abs(force[0]), abs(force[1]), abs(force[2])) <= zero_tolerance:
            continue
        rows.append((label, force[0], force[1], force[2]))

    return ('node', 'fx', 'fy', 'fz'), rows

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _open_csv_for_write(path):
    if sys.version_info[0] < 3:
        return open(path, 'wb')
    return open(path, 'w', newline='')

def _write_csv(path, header, rows):
    with _open_csv_for_write(path) as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))

def write_sample_csvs(output_root, sample_id, datasets):
    for dataset_name in ('coord', 'matrix', 'force', 'displace', 'stress'):
        if dataset_name not in datasets:
            raise KeyError('Dataset %r is missing.' % dataset_name)
        header, rows = datasets[dataset_name]
        subdir = DATASET_SUBDIRS[dataset_name]
        filename = '%s_%d.csv' % (dataset_name, int(sample_id))
        directory = os.path.join(output_root, subdir)
        ensure_dir(directory)
        _write_csv(os.path.join(directory, filename), header, rows)

# ---------------------------------------------------------------------------
# Sample processing / CLI
# ---------------------------------------------------------------------------

def discover_samples(input_dir):
    samples = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isdir(path):
            continue
        match = SAMPLE_PATTERN.match(name)
        if match:
            samples.append((int(match.group(1)), path))
    samples.sort(key=lambda item: item[0])
    return samples

def process_sample(sample_id, sample_dir, output_root, args):
    cae_path = os.path.join(sample_dir, 'connecting_lug_structure.cae')
    odb_path = os.path.join(sample_dir, 'connecting_lug_results.odb')
    if not os.path.isfile(cae_path):
        raise IOError('Missing file: %s' % cae_path)
    if not os.path.isfile(odb_path):
        raise IOError('Missing file: %s' % odb_path)

    odb = None
    opened_mdb = None
    try:
        odb = open_odb_read_only(odb_path)
        odb_instance = choose_odb_instance(odb, args.instance)
        odb_step_name, frame = choose_odb_step_and_frame(
            odb, requested_step=args.step, frame_index=args.frame
        )
        (
            node_labels,
            coord_by_label,
            coord_header,
            coord_rows,
            matrix_header,
            matrix_rows,
        ) = extract_mesh(odb_instance)

        displace_header, displace_rows = extract_displacement(
            frame, odb_instance, node_labels
        )
        stress_header, stress_rows = extract_stress(
            frame, odb_instance, node_labels
        )

        opened_mdb = openMdb(pathName=cae_path)
        model = choose_model(opened_mdb, odb_path, args.model)
        cae_step_name, cae_step = choose_cae_step(model, odb_step_name)
        force_header, force_rows = extract_forces(
            model=model,
            cae_step=cae_step,
            odb=odb,
            odb_instance=odb_instance,
            coord_by_label=coord_by_label,
            vertex_tolerance=args.vertex_tolerance,
            zero_tolerance=args.force_zero_tolerance,
        )

        datasets = {
            'coord': (coord_header, coord_rows),
            'matrix': (matrix_header, matrix_rows),
            'force': (force_header, force_rows),
            'displace': (displace_header, displace_rows),
            'stress': (stress_header, stress_rows),
        }
        write_sample_csvs(output_root, sample_id, datasets)

        force_sum = [0.0, 0.0, 0.0]
        for row in force_rows:
            force_sum[0] += float(row[1])
            force_sum[1] += float(row[2])
            force_sum[2] += float(row[3])

        return {
            'sample': sample_id,
            'instance': str(odb_instance.name),
            'odb_step': odb_step_name,
            'cae_step': cae_step_name,
            'nodes': len(node_labels),
            'elements': len(matrix_rows),
            'force_nodes': len(force_rows),
            'force_sum': tuple(force_sum),
        }
    finally:
        if opened_mdb is not None:
            try:
                opened_mdb.close()
            except Exception:
                try:
                    mdb.close()
                except Exception:
                    pass
        if odb is not None:
            try:
                odb.close()
            except Exception:
                pass

def build_argument_parser():
    parser = argparse.ArgumentParser(
        description='Export connecting_lug_i Abaqus samples to ML CSV files.'
    )
    parser.add_argument(
        '--input-dir',
        default='connecting_lug',
        help='Root containing connecting_lug_i folders. Default: ./connecting_lug',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output root. Default: sibling connecting_lug_processed.',
    )
    parser.add_argument(
        '--instance',
        default=None,
        help='ODB instance name. Default: choose the largest meshed instance.',
    )
    parser.add_argument(
        '--model',
        default=None,
        help='CAE model name. Required only if automatic model selection is ambiguous.',
    )
    parser.add_argument(
        '--step',
        default=None,
        help='ODB step name. Default: last ODB step.',
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=-1,
        help='Frame index in the selected step; negative indices are supported. Default: -1.',
    )
    parser.add_argument(
        '--vertex-tolerance',
        type=float,
        default=None,
        help=(
            'Maximum distance for CAE geometry/node coordinate -> ODB node '
            'fallback mapping. Default: automatic from model size.'
        ),
    )
    parser.add_argument(
        '--force-zero-tolerance',
        type=float,
        default=0.0,
        help='Treat force components with absolute value <= this as zero. Default: 0.',
    )
    parser.add_argument('--fail-fast', action='store_true')
    return parser

def _script_argv():
    """Return arguments intended for this script in Abaqus CAE noGUI mode."""
    argv = list(sys.argv[1:])
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    return argv

def main():
    parser = build_argument_parser()
    args, unknown = parser.parse_known_args(_script_argv())

    if unknown:
        log('[warning] Ignoring unrecognized arguments: %s' % ' '.join(unknown))

    if args.vertex_tolerance is not None and args.vertex_tolerance < 0.0:
        parser.error('--vertex-tolerance must be non-negative.')
    if args.force_zero_tolerance < 0.0:
        parser.error('--force-zero-tolerance must be non-negative.')

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise IOError('Input directory does not exist: %s' % input_dir)

    if args.output_dir is None:
        output_root = os.path.join(
            os.path.dirname(input_dir), 'connecting_lug_processed'
        )
    else:
        output_root = os.path.abspath(args.output_dir)

    for subdir in (
        'input_coord',
        'input_matrix',
        'input_force',
        'output_displace',
        'output_stress',
    ):
        ensure_dir(os.path.join(output_root, subdir))

    samples = discover_samples(input_dir)
    if not samples:
        raise RuntimeError(
            'No directories matching connecting_lug_<integer> were found in %s.'
            % input_dir
        )

    successes = []
    failures = []
    progress = tqdm(
        samples,
        total=len(samples),
        desc='Processing connecting lug samples',
        unit='sample',
        dynamic_ncols=True,
    )
    for sample_id, sample_dir in progress:
        progress.set_postfix(sample='connecting_lug_%d' % sample_id)
        try:
            summary = process_sample(
                sample_id, sample_dir, output_root, args
            )
            successes.append(summary)
            total = summary['force_sum']
            log(
                '[ok] connecting_lug_%d: instance=%s odb_step=%s cae_step=%s '
                'nodes=%d elements=%d force_nodes=%d '
                'sum(F)=(%.9g, %.9g, %.9g) N'
                % (
                    sample_id,
                    summary['instance'],
                    summary['odb_step'],
                    summary['cae_step'],
                    summary['nodes'],
                    summary['elements'],
                    summary['force_nodes'],
                    total[0], total[1], total[2],
                )
            )
        except Exception as exc:
            failures.append((sample_id, str(exc), traceback.format_exc()))
            log('[error] connecting_lug_%d failed: %s' % (sample_id, exc))
            if args.fail_fast:
                raise

    log('')
    log(
        'Exported %d/%d samples to %s'
        % (len(successes), len(samples), output_root)
    )
    if failures:
        for sample_id, message, trace in failures:
            log('connecting_lug_%d: %s' % (sample_id, message))
            log(trace)
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
