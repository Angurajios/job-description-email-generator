import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY"),
)


def get_job_description(url):
    loader = WebBaseLoader(url)
    page = loader.load().pop().page_content

    prompt_extract = PromptTemplate.from_template("""### SCRAPED TEXT FROM WEBSITE: {page_data}
    ### INSTRUCTION:
    The scraped text is from the career's page of a website.
    Your job is to extract the job postings and return them in JSON format containing
    the following keys: `role`, `experience`, `skills`, `company_name` and `description`.
    Only return the valid JSON.
    ### VALID JSON (NO PREAMBLE):""")

    chain = prompt_extract | llm | JsonOutputParser()
    return chain.invoke({"page_data": page})


def get_email(job):
    prompt_mail = PromptTemplate.from_template("""Write a detailed cold email to a recruiter applying for the role in the job description below.

    About me:
    - Name: Anguraj Selvam
    - I have 3 years of experience in software development and am moving into AI/ML.
    - I have completed AI/ML courses and am building projects in this area.
    - My CV is attached.

    Rules:
    - Read the job description and find the role title, company name, the main responsibilities, and the 3 or 4 most important skills it asks for.
    - Use the exact role and company name from the job description.
    - Write 3 short paragraphs plus a closing line:
    1. Intro: who I am, the role I am applying for, and where I saw it.
    2. Fit: connect my 3 years in software development (writing clean code, working in teams, testing, deploying, fixing production issues) to the main responsibilities in the job description.
    3. AI/ML: mention the 3 or 4 key skills from the job description and say that I have studied them through my AI/ML courses and am applying them in my own practice work. Say honestly that I am moving from software development into AI/ML and am keen to learn on the job.
    4. Closing: say my CV is attached, I am happy to do an assignment or interview.
    - Do not invent projects, company names, tools, numbers or achievements. Keep it general but specific to the job description.
    - Do not claim deep expertise. Be honest and confident.
    - 200 to 250 words, plain text, no bullet points.
    - Start with a subject line that includes the role title and my name.
    - Sound like a real person. Use simple words and short sentences. Vary sentence length a little.
    - Do not use words like "passionate", "excited", "leverage", "cutting-edge", "dynamic" or "I am writing to express my interest".
    - Do not use em dashes.
    - Output only the subject and the email.

    Job description:
    {job_description}""")

    chain_email = prompt_mail | llm
    res = chain_email.invoke({"job_description": str(job)})
    return res.content


st.set_page_config(page_title="Job Description to Cold Email", page_icon=":robot:")
st.title("Job Description to Cold Email")

url = st.text_input("Enter the URL of the job description page:")

if st.button("Generate Cold Email"):
    if not url:
        st.warning("Please enter a URL first.")
    else:
        with st.spinner("Generating..."):
            try:
                job = get_job_description(url)
                email = get_email(job)
                st.subheader("Generated Cold Email")
                st.text_area("Cold Email", value=email, height=300)
            except Exception as e:
                st.error(f"Something went wrong: {e}")