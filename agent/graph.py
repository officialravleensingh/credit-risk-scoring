from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes import risk_analyzer_node, regulation_retriever_node, report_generator_node


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node('risk_analyzer', risk_analyzer_node)
    builder.add_node('regulation_retriever', regulation_retriever_node)
    builder.add_node('report_generator', report_generator_node)

    builder.add_edge(START, 'risk_analyzer')
    builder.add_edge('risk_analyzer', 'regulation_retriever')
    builder.add_edge('regulation_retriever', 'report_generator')
    builder.add_edge('report_generator', END)

    return builder.compile()


lending_graph = build_graph()


def run_lending_agent(borrower_data: dict) -> dict:
    initial_state = {
        'borrower': borrower_data,
        'risk_summary': None,
        'retrieved_regulations': None,
        'final_report': None,
        'messages': []
    }
    result = lending_graph.invoke(initial_state)
    return {
        'risk_summary': result['risk_summary'],
        'retrieved_regulations': result['retrieved_regulations'],
        'final_report': result['final_report']
    }
