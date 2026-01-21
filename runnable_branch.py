from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

load_dotenv()

prompt1 = PromptTemplate(
  template="Write a detailed report about the topic: {topic}.",
  input_variables=["topic"],
)

prompt2 = PromptTemplate(
  template="Summarize the following report in a few sentences: {report}.",
  input_variables=["report"],
)

model = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

report_generator_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
  (lambda x : len(x.split()) > 100,prompt2 | model | parser),  
  RunnablePassthrough()
)

final_chain = report_generator_chain | branch_chain
print(final_chain.invoke({"topic": "Russia vs Ukraine war"}))