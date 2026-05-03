
def graph_details(graph, elem) -> dict[str, Any]:
    if isinstance(elem, tuple):
        return _edge_details(graph, elem)
    return _node_details(graph, elem)


def _details_callback(graph, elem):
    return graph_details(graph, elem)

def _pretty_print_node_schema(graph, node_id: int) -> None:
    print(f"[jedi_linear_nn_ir] node {node_id} schema:")
    pprint.pprint(graph_details(graph, node_id), sort_dicts=False, width=120)

def _pretty_print_edge_schema(graph, edge: tuple[int, int]) -> None:
    print(f"[jedi_linear_nn_ir] edge {edge} schema:")
    pprint.pprint(graph_details(graph, edge), sort_dicts=False, width=120)

