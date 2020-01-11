'''
Tests for dagviz and FlowImage class.
'''

import pytest
import io
import warnings
import unittest.mock as mock
import xml
from xml.etree import ElementTree as ET
import pydot
import networkx as nx
import PIL
import bionic.dagviz as dagviz


@pytest.fixture
def pydot_graph():
    '''Create empty pydot graph for testing'''
    return pydot.Dot(graph_type='graph')


@pytest.fixture
def flow_image(pydot_graph):
    '''Create FlowImage fixture for testing'''
    return dagviz.FlowImage(pydot_graph)


def test_flowimage_typing(flow_image):
    '''Check types for attributes of flowImage'''
    display_str = flow_image._repr_svg_()
    try:
        ET.fromstring(flow_image.xml_text)
    except xml.etree.ElementTree.ParseError:
        pytest.fail("FlowImage.xml_text not well formed XML")
    assert isinstance(flow_image, dagviz.FlowImage)
    assert isinstance(flow_image.pil_image, PIL.Image.Image)
    assert isinstance(flow_image.xml_text, bytes)
    assert isinstance(display_str, str)


@pytest.mark.parametrize("fp_input",
                         ['flow.svg', '/path/to/folder/flow.svg', 'filename.svg']
                         )
def test_save_flowimage_svg(fp_input, flow_image):
    '''Test custom saving for SVG files'''
    with mock.patch('builtins.open') as mock_open:
        flow_image.save(fp_input)
        assert mock_open.call_count == 1
        mock_open.assert_called_with(fp_input, 'wb')


def test_save_flowimage_warning(flow_image):
    '''Warning is thrown if svg image is saved with format options (not compatible'''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        flow_image.save('flow.svg', format='formatting options')
        assert len(w) == 1


@pytest.mark.parametrize("fp_input",
                         ['flow.png',
                          '/path/to/folder/flow.png',
                          'filename.png',
                          'machine_learning.jpeg'])
@mock.patch.object(PIL.Image.Image, 'save')
def test_save_flowimage_pil(mock_save, fp_input, flow_image):
    '''When the filepath extension is not svg, use PIL interface to save'''
    flow_image.save(fp_input)
    assert mock_save.call_count == 1


@mock.patch.object(PIL.Image.Image, 'save')
def test_save_flowimage_file_object(mock_save, flow_image):
    '''When a file object is given as input, use PIL interface to save'''
    file_object = io.TextIOWrapper(io.BytesIO())
    flow_image.save(file_object)
    assert mock_save.call_count == 1


@pytest.mark.parametrize("doc",
                         ['Hyperparams used to train model',
                          '\ttrain test split\n',
                          'fraüd modeling'
                          ])
def test_doc_propagated_to_tooltip(doc):
    '''Create a minimal networkx DiGraph and check that docs are propagated to tooltips'''
    G = nx.DiGraph()
    G.add_node(0, name='foo', doc=doc, task_ix=0, entity_name='buzz')
    dot = dagviz.dot_from_graph(G)
    assert isinstance(dot, pydot.Dot)
    assert dot.get_subgraphs()[0].get_nodes()[0].get_attributes()['tooltip'] == doc


def test_missing_doc_empty_tooltip():
    '''When doc is missing, tooltip is empty string'''
    G = nx.DiGraph()
    G.add_node(0, name='foo', doc=None, task_ix=0, entity_name='buzz')
    dot = dagviz.dot_from_graph(G)
    assert dot.get_subgraphs()[0].get_nodes()[0].get_attributes()['tooltip'] == ""
