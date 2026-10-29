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
    api_key=os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY"),
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


def get_email(job, profile):
    prompt_mail = PromptTemplate.from_template("""Write a cold email to a recruiter applying for the role in the job description below.

    About the applicant:
    - Name: {name}
    - Current background: {background}
    - Years of experience: {experience}
    - Extra notes: {notes}
    - CV attached: {cv_attached}

    Rules:
    - Read the job description and find the role title, company name, the main responsibilities, and the 3 or 4 most important skills it asks for.
    - Use the exact role and company name from the job description.
    - Never use placeholders in square brackets.
    - Write 3 short paragraphs plus a closing line:
    1. Intro: who the applicant is, the role they are applying for, and where they saw it.
    2. Fit: connect the applicant's background and experience to the main responsibilities in the job description.
    3. Skills: mention the 3 or 4 key skills the job asks for and say the applicant is interested in them and keen to learn and grow in them. Do not claim the applicant already has a skill unless it is stated in their background or notes.
    4. Closing: mention the CV if attached, and say they are happy to do an assignment or interview.
    - Do not invent projects, company names, tools, numbers or achievements. Use only the applicant details above.
    - Do not claim deep expertise. Be honest and confident.
    - {length} words, plain text, no bullet points.
    - Tone: {tone}.
    - Start with a subject line that includes the role title and the applicant's name.
    - Use simple words and short sentences. Sound like a real person.
    - Do not use words like "passionate", "excited", "leverage", "cutting-edge", "dynamic" or "I am writing to express my interest".
    - Do not use em dashes.
    - Output only the subject and the email.

    Job description:
    {job_description}""")

    chain_email = prompt_mail | llm
    res = chain_email.invoke({"job_description": str(job), **profile})
    return res.content


st.set_page_config(page_title="Job Description to Cold Email", page_icon=":robot:")
st.title("Job Description to Cold Email")
st.subheader("Your details")

col1, col2 = st.columns(2)
with col1:
    name = st.text_input("Your full name *")
    experience = st.text_input("Years of experience *", placeholder="e.g. 3 years")
with col2:
    background = st.text_input(
        "Current role or background *",
        placeholder="e.g. Software developer, fresher, MBA graduate",
    )
    tone = st.selectbox("Tone", ["Professional and friendly", "Formal", "Casual and warm"])

notes = st.text_input("Anything else to mention (optional)", placeholder="e.g. available to join in 30 days")
cv_attached = st.checkbox("I will attach my CV", value=True)
length = st.selectbox("Email length", ["150 to 200", "200 to 250", "250 to 300"], index=1)

st.divider()

url = st.text_input("Enter the URL of the job description page:")

if st.button("Generate Cold Email"):
    if not (name.strip() and experience.strip() and background.strip()):
        st.warning("Please fill in all the fields marked with *.")
    elif not url:
        st.warning("Please enter a URL first.")
    else:
        with st.spinner("Generating..."):
            try:
                job = get_job_description(url)
                if isinstance(job, list):
                    job = job[0]
                if not job:
                    st.error("Could not extract job details from this page.")
                else:
                    profile = {
                        "name": name,
                        "background": background,
                        "experience": experience,
                        "notes": notes or "None",
                        "cv_attached": "Yes" if cv_attached else "No",
                        "tone": tone,
                        "length": length,
                    }
                    email = get_email(job, profile)
                    st.subheader("Generated Cold Email")
                    st.text_area("Cold Email", value=email, height=300)
            except Exception as e:
                st.error(f"Something went wrong: {e}")