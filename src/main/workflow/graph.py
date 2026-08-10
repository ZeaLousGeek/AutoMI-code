from langgraph.graph import StateGraph, END
from src.main.workflow.state import AgentState, initialize_state
from src.main.workflow.planning import planning_node
from src.main.workflow.execution import execution_node
from src.main.workflow.feedback import feedback_node, should_continue
from src.main.workflow.output import output_node


def create_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node('planning', planning_node)
    workflow.add_node('execution', execution_node)
    workflow.add_node('feedback', feedback_node)
    workflow.add_node('output', output_node)

    workflow.set_entry_point('planning')

    workflow.add_edge('planning', 'execution')
    workflow.add_edge('execution', 'feedback')
    workflow.add_conditional_edges(
        'feedback',
        should_continue,
        {
            'planning': 'planning',
            'output': 'output'
        }
    )
    workflow.add_edge('output', END)

    return workflow.compile()


def run_automi_system(max_iterations: int = 10, selected_model_name=None,
                      selected_model_path=None, test_mode=False,
                      selected_datasets=None,
                      max_consecutive_param_failures=3,
                      dataset_name=None, subject_id=None,
                      base_output_dir=None):
    tag = ""
    if dataset_name and subject_id:
        tag = f"[{dataset_name}/Sub{subject_id}] "

    print("=" * 60)
    print(f"{tag}AutoMI 运动想象脑电信号分类模型自动迭代系统")
    print("=" * 60)

    app = create_workflow()
    initial_state = initialize_state(
        max_iterations, selected_model_name, selected_model_path, test_mode,
        selected_datasets,
        max_consecutive_param_failures=max_consecutive_param_failures,
        dataset_name=dataset_name,
        subject_id=subject_id,
        base_output_dir=base_output_dir,
    )

    result = app.invoke(initial_state)

    return result


if __name__ == '__main__':
    run_automi_system(max_iterations=5)
