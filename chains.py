import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Vipul Singh, a student at IIT (ISM),DHanbad, pusuing M.Tech in Industrial Engineering & management.
            Your course consists of courses like is a mixture of data science (like Business Analytics, Machine Learning and AI), Process and
            quality improvement (like Quality management, Productivity management), Operation research and management 
            (like Decsion modeling, Project management, Stochastic Programming, Supply Chain management).
            You have done internship of 5 months in Business Excellence department, where you utilized your skills in Operations research and management and also got hands on experience on Lean Six Sigma methodology.
            You also assisted the COO of the company through data analytics and reports using PowerBI.      
            You are a KPMG certified Lean Six Sigma Green Belt. And have won first place in a strategy competition at Unstop using these skills.
            Over your experience, you have performed process optimization, cost reduction, and heightened overall efficiency. 
            
            Your job is to write a cold email to the hiring manager regarding the job mentioned above describing the capability and how good of a fit you are
            in fulfilling their needs.
            Also add the most relevant ones from the following links to showcase your portfolio: {link_list}
            Remember you are Vipu Singh, M.Tech (IEM) at IIT (ISM) Dhanbad.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))