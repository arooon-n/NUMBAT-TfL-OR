import pandas as pd
from src.ingestion.loader import detect_schema
from src.preprocessing.clean import canonicalize
from src.network.graph import build_graph
from src.models.optimization import analytical_required_frequency


def test_schema_detection():
    schema=detect_schema(["Line", "From Station", "To Station", "Link Load", "Link Frequency"])
    assert schema["link_load"] == "Link Load"


def test_clean_rejects_negative():
    frame=pd.DataFrame({"Line":["A"],"From Station":["X"],"To Station":["Y"],"Link Load":[-1],"Link Frequency":[2]})
    schema=detect_schema(list(frame.columns))
    try: canonicalize(frame,schema)
    except ValueError: return
    assert False


def test_network():
    frame=pd.DataFrame({"from_station":["A"],"to_station":["B"],"line":["L"],"link_load":[2],"link_frequency":[1]})
    assert build_graph(frame).number_of_edges()==1


def test_analytical_frequency():
    frame=pd.DataFrame({"line":["L"],"service_group":["L"],"time_period":["08:00"],"link_load":[101],"link_frequency":[1]})
    capacity=pd.DataFrame({"line":["L"],"service_group":["L"],"capacity_per_train":[100]})
    assert analytical_required_frequency(frame,capacity).required_frequency.iloc[0] == 2
