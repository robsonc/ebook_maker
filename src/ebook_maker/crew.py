from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from crewai_tools import SerperDevTool
#import agentops

# Uncomment the following line to use an example of a custom tool
# from ebook_maker.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool
search_tool = SerperDevTool()

#agentops.init()
llm = ChatOpenAI(model="gpt-4o-2024-05-13")

groq = ChatGroq(
    temperature=0.7, 
    groq_api_key = "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", 
    # model_name="llama3-70b-8192",
    # model_name="llama3-8b-8192",
    # model_name="mixtral-8x7b-32768",
    #max_tokens=3000
)


@CrewBase
class EbookMakerCrew():
	"""EbookMaker crew"""
	agents_config = 'config/agents-ebook-maker.yaml'
	tasks_config = 'config/tasks-ebook-maker.yaml'
 
	# def project_editor(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['project_editor'],
	# 		verbose=False,
	# 		allow_delegation=True,
	# 		llm=llm
	# 	)

	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			tools=[search_tool],
			verbose=False
		)
  
	@agent
	def developmental_editor(self) -> Agent:
		return Agent(
			config=self.agents_config['developmental_editor'],
			verbose=False
		)
  
	@agent
	def content_writer(self) -> Agent:
		return Agent(
			config=self.agents_config['content_writer'],
			verbose=False,
			llm=groq
		)
  
	# @task
	# def project_editing_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['project_editing_task'],
	# 		agent=self.project_editor()
	# 	)

	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
			agent=self.researcher()
		)
  
	@task
	def developmental_editing_task(self) -> Task:
		return Task(
			config=self.tasks_config['developmental_editing_task'],
			agent=self.developmental_editor(),
			output_file='outline.md'
		)

	@task
	def content_writing_task(self) -> Task:
		return Task(
			config=self.tasks_config['content_writing_task'],
			agent=self.content_writer(),
			output_file='content.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the EbookMaker crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.hierarchical,
			#manager_agent=self.project_editor(),
			manager_llm=llm,
			verbose=2,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)