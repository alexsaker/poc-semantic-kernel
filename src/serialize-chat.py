# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.core_plugins import ConversationSummaryPlugin
from semantic_kernel.prompt_template.input_variable import InputVariable
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextCompletion,
)


def add_service(kernel,use_chat,service_id):
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
    # Initialize the kernel
    kernel = sk.Kernel()

    service_id = "default"
    script_directory = os.path.dirname(__file__)
    plugins_directory = os.path.join(script_directory,".." ,"plugins")
    summarize_functions = kernel.add_plugin(parent_directory=plugins_directory, plugin_name="SummarizePlugin")
    topics_function = summarize_functions["Topics"]
    #topics_function = summarize_functions["Topics"]
    topics_function = kernel.get_function(
        plugin_name="SummarizePlugin", function_name="Topics"
    )
    # Add the service to the kernel
    # use_chat: True to use chat completion, False to use text completion
    kernel = add_service(kernel=kernel, use_chat=True,service_id=service_id)

    execution_settings = PromptExecutionSettings(
        service_id=service_id, max_tokens=ConversationSummaryPlugin._max_tokens, temperature=0.1, top_p=0.5
    )

    template = (
        "BEGIN CONTENT TO SUMMARIZE:\n{{" + "$history" + "}}\n{{" + "$input" + "}}\n"
        "END CONTENT TO SUMMARIZE.\nSummarize the conversation in 'CONTENT TO"
        " SUMMARIZE',            identifying main points of discussion and any"
        " conclusions that were reached.\nDo not incorporate other general"
        " knowledge.\nSummary is in plain text, in complete sentences, with no markup"
        " or tags.\n\nBEGIN SUMMARY:\n"
    )

    prompt_template_config = PromptTemplateConfig(
        template=template,
        description="Given a section of a conversation transcript, summarize the part of the conversation.",
        execution_settings=execution_settings,
        InputVariables=[
            InputVariable(name="input", description="The user input", is_required=True),
            InputVariable(name="history", description="The history of the conversation", is_required=True),
        ],
    )

    # Import the ConversationSummaryPlugin
    kernel.add_plugin(
        ConversationSummaryPlugin( prompt_template_config=prompt_template_config),
        plugin_name="ConversationSummaryPlugin",
    )

    summarize_function = kernel.get_function(
        plugin_name="ConversationSummaryPlugin", function_name="SummarizeConversation"
    )
   

    # Create the history
    history = ChatHistory()

    while True:
        try:
            request = input("User:> ")
        except KeyboardInterrupt:
            print("\n\nExiting chat...")
            return False
        except EOFError:
            print("\n\nExiting chat...")
            return False

        if request == "exit":
            print("\n\nExiting chat...")
            return False

        result = await kernel.invoke(
            summarize_function,
            input=request,
            history=history,
        )
        result_topics = await kernel.invoke(
            topics_function,
            input=request,
            history=history,
        )

        # Add the request to the history
        history.add_user_message(request)
        history.add_assistant_message(str(result))
        history.add_assistant_message(str(result_topics))

        print(f"Assistant:> {result}\nTOPICS: {result_topics}")


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
