# Copyright (c) Microsoft. All rights reserved.
import asyncio
import os
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextCompletion,
)
from semantic_kernel.contents import (
    ChatHistory,
)
from semantic_kernel import Kernel
from semantic_kernel.planners.function_calling_stepwise_planner import FunctionCallingStepwisePlanner

def add_service(kernel,use_chat):
    service_id="planner"
    if use_chat:
        # <OpenAIKernelCreation>
        kernel.add_service(OpenAIChatCompletion(service_id=service_id))
        # </OpenAIKernelCreation>
    else:
        # <OpenAITextCompletionKernelCreation>
        kernel.add_service(OpenAITextCompletion(service_id=service_id))
        # </OpenAITextCompletionKernelCreation>
    return kernel

async def main():
    # <CreatePlanner>
    # Initialize the kernel
    kernel = Kernel()

    # Add the service to the kernel
    # use_chat: True to use chat completion, False to use text completion
   
    kernel = add_service(kernel=kernel, use_chat=True)

    script_directory = os.path.dirname(__file__)
    plugins_directory = os.path.join(script_directory, "..","plugins")
    kernel.add_plugin(parent_directory=plugins_directory, plugin_name="MathPlugin")

    planner = FunctionCallingStepwisePlanner(service_id="default")
    # </CreatePlanner>
    # <RunPlanner>
    chat_history = ChatHistory()
    user_input_example = "Figure out how much I have if first, my investment of 2130.23 dollars increased by 23%, and then I spend $5 on a coffee"
    
    while True:
        try:
            user_input = input("What is your mathematical problem? ")
        except Exception:
            break
        if user_input == "exit":
            break
        if not user_input:
            user_input = user_input_example

       
        goal = user_input  # noqa: E501
        chat_history.add_user_message(user_input)
        # Execute the plan
        result = await planner.invoke(kernel=kernel, question=chat_history.to_prompt())
        chat_history.add_assistant_message(result.final_answer)
        print(f"The goal: {goal}")
        print(f"Global result: {result}")
        print(f"Plan result: {result.final_answer}")
    print("Thanks for chatting with me!")
    # </RunPlanner>


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
