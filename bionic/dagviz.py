"""
Contains code for visualizing Bionic flow graphs.  Importing this module will
pull in several optional dependencies as well.  The external Graphviz library
is also required.
"""

import warnings
from pathlib import Path
from collections import defaultdict
from io import BytesIO, IOBase

from .optdep import import_optional_dependency, oneline
module_purpose = 'rendering the flow DAG'
hsluv = import_optional_dependency('hsluv', purpose=module_purpose)
pydot = import_optional_dependency('pydot', purpose=module_purpose)
Image = import_optional_dependency('PIL.Image', purpose=module_purpose)


class FlowImage:
    def __init__(self, pydot_graph):
        '''
        Given a pydot graph object, create Pillow Image or SVG represented as XML text.
        '''
        self.pil_image = Image.open(BytesIO(pydot_graph.create_png()))
        self.xml_text = pydot_graph.create_svg()

    def save(self, fp, format=None, **params):
        '''Save flow visualization under the given filename. For SVG, must pass a filepath and does not
        support file objects. For formats supported by PIL, pass a filename, path or file object,
        and optionally specify format, else infer from filename extension. Pass additonal keyword
        options supported by PIL using params.'''
        if isinstance(fp, IOBase):
            # write to file object using PIL interface
            self.pil_image.save(fp, format, **params)
        else:
            if Path(fp).suffix == '.svg':
                if format or params:
                    warnings.warn(oneline('''Optional args ``format`` and ``params`` passed to FlowImage.save() are
                       not supported for SVG.'''))
                with open(fp, 'wb') as file:
                    file.write(self.xml_text)
            else:
                self.pil_image.save(fp, format, **params)

    def show(self):
        '''Show image using PIL'''
        self.pil_image.show()

    def _repr_svg_(self):
        '''Rich display image as SVG in IPython notebook or Qt console.'''
        return str(self.xml_text)


def hpluv_color_dict(keys, saturation, lightness):
    '''
    Given a list of arbitary keys, generates a dict mapping those keys to a set
    of evenly-spaced, perceptually uniform colors with the specified saturation
    and lightness.
    '''

    n = len(keys)
    color_strs = [
        hsluv.hpluv_to_hex([(360 * (i / float(n))), saturation, lightness])
        for i in range(n)
    ]
    return dict(zip(keys, color_strs))


def dot_from_graph(graph, vertical=False, curvy_lines=False):
    '''
    Given a NetworkX directed acyclic graph, returns a Pydot object which can
    be visualized using GraphViz.
    '''

    dot = pydot.Dot(
        graph_type='digraph',
        splines='spline' if curvy_lines else 'line',
        outputorder='edgesfirst',
        rankdir='TB' if vertical else 'LR',
    )

    node_lists_by_cluster = defaultdict(list)
    for node in graph.nodes():
        entity_name = graph.nodes[node]['entity_name']
        node_lists_by_cluster[entity_name].append(node)

    entity_names = list(set(
        graph.nodes[node]['entity_name']
        for node in graph.nodes()
    ))
    color_strs_by_entity_name = hpluv_color_dict(
        entity_names, saturation=99, lightness=90)

    def name_from_node(node):
        return graph.nodes[node]['name']

    def doc_from_node(node):
        doc = graph.nodes[node]['doc']
        return doc if doc else ""

    for cluster, node_list in node_lists_by_cluster.items():
        sorted_nodes = list(sorted(
            node_list, key=lambda node: graph.nodes[node]['task_ix']))

        subdot = pydot.Cluster(cluster, style='invis')

        for node in sorted_nodes:
            entity_name = graph.nodes[node]['entity_name']
            subdot.add_node(pydot.Node(
                name_from_node(node),
                tooltip=doc_from_node(node),
                style='filled',
                fillcolor=color_strs_by_entity_name[entity_name],
                shape='box',
            ))

        dot.add_subgraph(subdot)

    for pred_node in graph.nodes():
        for succ_node in graph.successors(pred_node):
            dot.add_edge(pydot.Edge(
                name_from_node(pred_node),
                name_from_node(succ_node),
                arrowhead='open',
                tailport='s' if vertical else 'e',
            ))

    return dot
