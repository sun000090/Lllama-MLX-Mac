from src.agent_executors import AgentExec
from logs.logger import Logger
import os

logss = Logger().get_logger('Logger')

class InferenceAgent:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(InferenceAgent, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance():
        if InferenceAgent._instance is None:
            InferenceAgent()
        return InferenceAgent._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.agent_executors = self.AgentExecutor()  # Initialize once
            self.initialized = True

    def AgentExecutor(self):
        agent_executors = AgentExec(loc_name=os.getenv('CONF_DIR'))
        agent_executors_ = agent_executors.agent_executor()
        logss.info("Agent executor loaded")
        return agent_executors_

    def inferences(self, input_prompt):
        try:
            agents = self.agent_executors  # Use the initialized instance
            events = agents.stream(
                {"messages": [("user", input_prompt)]},
                stream_mode="values"
            )
            
            messages = []
            for event in events:
                messages.append(event["messages"][-1])
                logss.info(event["messages"])
            final_messages = messages[-1].content
            final_table = messages[-2].content
            final_query = messages[-3].additional_kwargs['tool_calls'][0]['function']['arguments']
            return final_messages, final_table, final_query
        except Exception as e:
            logss.error(f"Error in dynamic execution: {e}")
            return None, None, None
