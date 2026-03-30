import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os


import requests

API_URL = "https://router.huggingface.co/models/distilbert/distilbart-cnn-12-6"
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def get_ai_feedback(resume_text, job_role):
    prompt = f"""
    You are an ATS system.

    Analyze this resume for the role of {job_role}.

    Give:
    1. Strengths
    2. Weaknesses
    3. Missing skills
    4. Improvement suggestions

    Resume:
    {resume_text[:1000]}
    """

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "temperature": 0.7
                }
            }
        )


        if response.status_code != 200:
            return f"API Error {response.status_code}: {response.text}"

        # SAFE JSON PARSE
        try:
            result = response.json()
        except:
            return f"Invalid JSON response: {response.text}"

        # HANDLE HF LOADING
        if isinstance(result, dict) and "error" in result:
            return f"Model Issue: {result['error']}"

        # HANDLE SUCCESS
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No output generated")

        return "Unexpected response format"

    except Exception as e:
        return f"Request failed: {str(e)}"
def extract_text(file):
    reader=PdfReader(file)
    text=""
    for page in reader.pages:
        content=page.extract_text()
        if content:
            text=text+content
    return text
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
def generate_smart_feedback(score, skill_score, missing_skills, resume_text):
    feedback = []


    if score < 50:
        feedback.append("Resume is poorly aligned with the target role.")
    elif score < 75:
        feedback.append("Resume is moderately aligned. Improvements needed.")
    else:
        feedback.append("Resume is strongly aligned with the target role.")


    if skill_score < 40:
        feedback.append("Critical skill gap detected. You lack core required skills.")
    elif skill_score < 70:
        feedback.append("You have some relevant skills but need improvement.")


    if missing_skills:
        feedback.append(f"Missing key skills: {', '.join(missing_skills[:5])}")

    # Sections check
    if "project" not in resume_text:
        feedback.append("Add 2–3 strong projects to improve impact.")

    if "experience" not in resume_text:
        feedback.append("Add experience or internships section.")

    if "skills" not in resume_text:
        feedback.append("Add a dedicated skills section.")


    word_count = len(resume_text.split())
    if word_count < 300:
        feedback.append("Resume is too short. Add more technical depth.")
    elif word_count > 1200:
        feedback.append("Resume is too long. Keep it concise (1 page ideal).")

    return feedback




def calculate_similarity(model,resume_text, job_text):
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_text, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2)
    return float(similarity[0][0]) * 100


def skill_match_score(resume_text, skills_required):
    matched = [skill for skill in skills_required if skill in resume_text]
    score = (len(matched) / len(skills_required)) * 100 if skills_required else 0
    return score, matched

job_data = {

# ---------------- ENGINEERING ----------------

"Software Developer": {
    "description": "Develops, tests, and maintains software applications and systems.",
    "skills": ["python", "java", "c++", "data structures", "algorithms", "git", "apis"],
    "keywords": ["oop", "debugging", "backend", "frontend", "software development"]
},

"Data Scientist": {
    "description": "Analyzes data using statistical methods and machine learning techniques.",
    "skills": ["python", "machine learning", "sql", "statistics", "pandas", "numpy"],
    "keywords": ["data analysis", "ml", "regression", "classification", "nlp"]
},

"Cyber Security Analyst": {
    "description": "Protects systems and networks from cyber threats and vulnerabilities.",
    "skills": ["networking", "linux", "siem", "penetration testing", "firewalls"],
    "keywords": ["soc", "threat detection", "owasp", "mitre", "ids", "ips"]
},

"Embedded Systems Engineer": {
    "description": "Designs and develops embedded hardware and software systems.",
    "skills": ["c", "c++", "microcontrollers", "embedded systems", "rtos"],
    "keywords": ["firmware", "arduino", "hardware interfacing", "sensors"]
},

"IoT Engineer": {
    "description": "Builds and manages connected systems using sensors and cloud technologies.",
    "skills": ["iot", "embedded systems", "python", "cloud computing", "networking"],
    "keywords": ["mqtt", "edge computing", "sensors", "automation"]
},

"VLSI Engineer": {
    "description": "Designs integrated circuits and semiconductor devices.",
    "skills": ["verilog", "vhdl", "digital electronics", "circuit design"],
    "keywords": ["asic", "fpga", "rtl design", "semiconductors"]
},

"Design Engineer": {
    "description": "Designs mechanical components using CAD tools.",
    "skills": ["autocad", "solidworks", "mechanics", "material science"],
    "keywords": ["cad", "product design", "simulation"]
},

"Automotive Engineer": {
    "description": "Develops and improves automotive systems and vehicles.",
    "skills": ["mechanics", "vehicle dynamics", "thermodynamics"],
    "keywords": ["engine design", "automotive systems"]
},

"Production Engineer": {
    "description": "Optimizes manufacturing processes for efficiency and quality.",
    "skills": ["manufacturing", "quality control", "process optimization"],
    "keywords": ["lean manufacturing", "six sigma", "production"]
},

"Structural Engineer": {
    "description": "Designs and analyzes structures such as buildings and bridges.",
    "skills": ["structural analysis", "civil engineering", "autocad"],
    "keywords": ["load analysis", "steel design", "concrete"]
},

"Site Engineer": {
    "description": "Supervises construction work on-site and ensures project execution.",
    "skills": ["construction", "project management", "site supervision"],
    "keywords": ["safety", "planning", "execution"]
},

"Construction Manager": {
    "description": "Manages construction projects, budgets, and timelines.",
    "skills": ["project management", "budgeting", "construction"],
    "keywords": ["scheduling", "leadership", "planning"]
},

"Power Systems Engineer": {
    "description": "Works on electrical power generation and distribution systems.",
    "skills": ["power systems", "electrical engineering", "matlab"],
    "keywords": ["grid", "substation", "power flow"]
},

"Control Systems Engineer": {
    "description": "Designs automated control systems for machines and processes.",
    "skills": ["control systems", "matlab", "automation"],
    "keywords": ["pid controller", "robotics", "feedback systems"]
},

"Electrical Design Engineer": {
    "description": "Designs electrical circuits and systems.",
    "skills": ["circuit design", "pcb design", "electrical engineering"],
    "keywords": ["schematics", "simulation", "matlab"]
},

# ---------------- MANAGEMENT ----------------

"Digital Marketing Manager": {
    "description": "Manages digital marketing campaigns and online branding.",
    "skills": ["seo", "sem", "google ads", "analytics"],
    "keywords": ["campaigns", "marketing strategy", "traffic"]
},

"SEO Specialist": {
    "description": "Optimizes websites to rank higher on search engines.",
    "skills": ["seo", "keyword research", "analytics"],
    "keywords": ["backlinks", "on-page seo", "ranking"]
},

"Brand Manager": {
    "description": "Develops and maintains brand identity and strategy.",
    "skills": ["branding", "marketing", "communication"],
    "keywords": ["brand strategy", "campaigns", "positioning"]
},

"Financial Analyst": {
    "description": "Analyzes financial data for business decision-making.",
    "skills": ["financial modeling", "excel", "accounting"],
    "keywords": ["forecasting", "valuation", "analysis"]
},

"Investment Banker": {
    "description": "Handles financial investments, mergers, and acquisitions.",
    "skills": ["finance", "valuation", "financial modeling"],
    "keywords": ["ipo", "m&a", "capital markets"]
},

"Risk Analyst": {
    "description": "Identifies and manages financial and operational risks.",
    "skills": ["risk management", "finance", "analysis"],
    "keywords": ["compliance", "risk assessment"]
},

"HR Manager": {
    "description": "Manages recruitment, employee relations, and HR policies.",
    "skills": ["recruitment", "hr management", "communication"],
    "keywords": ["payroll", "policies", "employee relations"]
},

"Talent Acquisition Specialist": {
    "description": "Handles hiring and recruitment processes.",
    "skills": ["recruitment", "screening", "communication"],
    "keywords": ["hiring", "interviews", "ats"]
},

"HR Analyst": {
    "description": "Analyzes HR data to improve workforce decisions.",
    "skills": ["data analysis", "hr analytics", "excel"],
    "keywords": ["reporting", "metrics", "analysis"]
},

"Business Analyst": {
    "description": "Analyzes business needs and provides solutions.",
    "skills": ["sql", "excel", "communication"],
    "keywords": ["requirement gathering", "analysis", "documentation"]
},

"Data Analyst": {
    "description": "Processes and analyzes data for insights.",
    "skills": ["sql", "excel", "power bi", "python"],
    "keywords": ["data visualization", "analysis", "reporting"]
},

"Strategy Consultant": {
    "description": "Advises organizations on business strategies.",
    "skills": ["analysis", "problem solving", "communication"],
    "keywords": ["strategy", "consulting", "market analysis"]
},

# ---------------- HEALTHCARE ----------------

"Doctor": {
    "description": "Diagnoses and treats patients.",
    "skills": ["medical knowledge", "diagnosis", "patient care"],
    "keywords": ["treatment", "healthcare"]
},

"Surgeon": {
    "description": "Performs surgical procedures.",
    "skills": ["surgery", "medical procedures", "precision"],
    "keywords": ["operations", "surgical care"]
},

"Medical Researcher": {
    "description": "Conducts research in medical and healthcare fields.",
    "skills": ["research", "data analysis", "clinical trials"],
    "keywords": ["study", "experiments", "healthcare research"]
},

"Pharmacist": {
    "description": "Dispenses medicines and advises patients.",
    "skills": ["pharmacy", "medicines", "patient care"],
    "keywords": ["prescriptions", "drug safety"]
},

"Clinical Pharmacist": {
    "description": "Works with doctors to optimize drug therapy.",
    "skills": ["pharmacology", "clinical knowledge"],
    "keywords": ["drug therapy", "patient care"]
},

"Drug Safety Associate": {
    "description": "Monitors drug safety and side effects.",
    "skills": ["pharmacovigilance", "data analysis"],
    "keywords": ["drug safety", "monitoring"]
},

"Staff Nurse": {
    "description": "Provides basic patient care.",
    "skills": ["nursing", "patient care"],
    "keywords": ["monitoring", "healthcare"]
},

"ICU Nurse": {
    "description": "Handles critical care patients.",
    "skills": ["critical care", "nursing"],
    "keywords": ["icu", "monitoring"]
},

"Nursing Supervisor": {
    "description": "Manages nursing staff and operations.",
    "skills": ["management", "nursing"],
    "keywords": ["supervision", "healthcare"]
},

"Public Health Officer": {
    "description": "Works on public health policies and programs.",
    "skills": ["public health", "policy"],
    "keywords": ["community health", "programs"]
},

"Epidemiologist": {
    "description": "Studies disease patterns.",
    "skills": ["data analysis", "research"],
    "keywords": ["disease study", "statistics"]
},

"Health Program Manager": {
    "description": "Manages healthcare programs.",
    "skills": ["management", "healthcare"],
    "keywords": ["program management", "planning"]
},

# ---------------- DESIGN ----------------

"Graphic Designer": {
    "description": "Creates visual content using design tools.",
    "skills": ["photoshop", "illustrator", "creativity"],
    "keywords": ["graphics", "design", "visuals"]
},

"Brand Designer": {
    "description": "Designs brand identity and visuals.",
    "skills": ["branding", "design"],
    "keywords": ["identity", "visual design"]
},

"Visual Designer": {
    "description": "Creates visual elements for digital platforms.",
    "skills": ["ui design", "graphics"],
    "keywords": ["layout", "visuals"]
},

"UI Designer": {
    "description": "Designs user interfaces for applications.",
    "skills": ["figma", "ui design"],
    "keywords": ["interfaces", "design"]
},

"UX Researcher": {
    "description": "Studies user behavior to improve products.",
    "skills": ["user research", "analysis"],
    "keywords": ["usability", "testing"]
},

"Product Designer": {
    "description": "Designs user-centric products.",
    "skills": ["ui ux", "design thinking"],
    "keywords": ["product design", "innovation"]
},

"Design Strategist": {
    "description": "Aligns design with business strategy.",
    "skills": ["strategy", "design"],
    "keywords": ["innovation", "planning"]
},

"Interaction Designer": {
    "description": "Designs user interactions in digital products.",
    "skills": ["interaction design", "prototyping"],
    "keywords": ["ux", "interfaces"]
},

"Animator": {
    "description": "Creates animations and motion visuals.",
    "skills": ["animation", "motion design"],
    "keywords": ["visual effects", "graphics"]
},

"Motion Graphics Designer": {
    "description": "Designs animated graphics for media.",
    "skills": ["after effects", "animation"],
    "keywords": ["motion graphics", "visuals"]
},

"3D Artist": {
    "description": "Creates 3D models and renders.",
    "skills": ["blender", "3d modeling"],
    "keywords": ["rendering", "visualization"]
}

}


branch ="select"
target = "select"

st.title("ATS RESUME EVALUATOR")
st.subheader("Made by HAMZA AND NIMISH")

st.write("GIVE ME A RESUME")

st.sidebar.write(
"Unlock the full power of our ATS Resume Evaluator with the Premium Membership. "
"For just ₹999, you get unlimited resume uploads, advanced ATS scoring, keyword analysis, "
"and personalized suggestions to improve your resume and increase your chances of getting shortlisted."
)

with st.expander("NOTE"):
    st.write("This Website is a concept by the creators and will be equipped with more ideas and helpful tools required for a student's training and placement.")


res = st.radio(
    "Select the domain you intend to work in or are targeting currently.",
    [
        "Engineering",
        "Management",
        "Healthcare",
        "Design"
    ]
)

# ---------------- ENGINEERING ----------------

if res == "Engineering":

    branch = st.selectbox("Select Branch :", [
        "Select",
        "Computer Science",
        "Electronics",
        "Mechanical",
        "Civil",
        "Electrical"
    ])

    if branch == "Computer Science":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Software Developer",
            "Data Scientist",
            "Cyber Security Analyst"
        ])

    elif branch == "Electronics":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Embedded Systems Engineer",
            "IoT Engineer",
            "VLSI Engineer"
        ])

    elif branch == "Mechanical":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Design Engineer",
            "Automotive Engineer",
            "Production Engineer"
        ])

    elif branch == "Civil":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Structural Engineer",
            "Site Engineer",
            "Construction Manager"
        ])

    elif branch == "Electrical":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Power Systems Engineer",
            "Control Systems Engineer",
            "Electrical Design Engineer"
        ])

# ---------------- MANAGEMENT ----------------

elif res == "Management":

    branch = st.selectbox("Select Branch :", [
        "Select",
        "Marketing",
        "Finance",
        "Human Resources",
        "Business Analytics"
    ])

    if branch == "Marketing":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Digital Marketing Manager",
            "SEO Specialist",
            "Brand Manager"
        ])

    elif branch == "Finance":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Financial Analyst",
            "Investment Banker",
            "Risk Analyst"
        ])

    elif branch == "Human Resources":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "HR Manager",
            "Talent Acquisition Specialist",
            "HR Analyst"
        ])

    elif branch == "Business Analytics":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Business Analyst",
            "Data Analyst",
            "Strategy Consultant"
        ])

# ---------------- HEALTHCARE ----------------

elif res == "Healthcare":

    branch = st.selectbox("Select Branch :", [
        "Select",
        "Medicine",
        "Pharmacy",
        "Nursing",
        "Public Health"
    ])

    if branch == "Medicine":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Doctor",
            "Surgeon",
            "Medical Researcher"
        ])

    elif branch == "Pharmacy":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Pharmacist",
            "Clinical Pharmacist",
            "Drug Safety Associate"
        ])

    elif branch == "Nursing":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Staff Nurse",
            "ICU Nurse",
            "Nursing Supervisor"
        ])

    elif branch == "Public Health":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Public Health Officer",
            "Epidemiologist",
            "Health Program Manager"
        ])

# ---------------- DESIGN ----------------

elif res == "Design":

    branch = st.selectbox("Select Branch :", [
        "Select",
        "Graphic Design",
        "UI/UX Design",
        "Product Design",
        "Animation"
    ])

    if branch == "Graphic Design":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Graphic Designer",
            "Brand Designer",
            "Visual Designer"
        ])

    elif branch == "UI/UX Design":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "UI Designer",
            "UX Researcher",
            "Product Designer"
        ])

    elif branch == "Product Design":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Product Designer",
            "Design Strategist",
            "Interaction Designer"
        ])

    elif branch == "Animation":
        target = st.selectbox("Select Target Role :", [
            "Select",
            "Animator",
            "Motion Graphics Designer",
            "3D Artist"
        ])



if branch != "Select" and target != "Select":

    st.divider()
    st.subheader(f"Analyzing for: {target}")

    file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

    if file is not None:
        with st.spinner("Uploading resume..."):
            model = load_model()
        resume_text = extract_text(file)
        resume_text = resume_text.replace("\n", " ").lower()[:3000]


        role_info = job_data.get(target, {})

        if not role_info:
            st.warning("Role data not found")
        else:
            job_desc = role_info.get("description", "")
            skills_required = role_info.get("skills", [])
            keywords_required = role_info.get("keywords", [])


            job_text = job_desc + " " + " ".join(skills_required) + " " + " ".join(keywords_required)


            bert_score = calculate_similarity(model,resume_text, job_text)
            skill_score, matched_skills = skill_match_score(resume_text, skills_required)
            missing_skills = [skill for skill in skills_required if skill.lower() not in resume_text.lower()]
            final_score = (0.6 * bert_score) + (0.4 * skill_score)

            smart_feedback = generate_smart_feedback(
                final_score,
                skill_score,
                missing_skills,
                resume_text
            )


            st.metric("BERT Score", f"{bert_score:.2f}%")
            st.metric("Skill Match", f"{skill_score:.2f}%")
            st.metric("Final ATS Score", f"{final_score:.2f}%")
            st.subheader("Smart Feedback")

            for point in smart_feedback:
                st.write(f"• {point}")





            st.subheader("Missing Skills")
            st.write(missing_skills)










# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown(
"""
<div style='text-align:center; font-size:17px;'>
© 2026 ATS Resume Evaluator | Made by NIMISH And TEAM
    <div style='text-align:center; font-size:17px;'>  
     
        Email : nileshkrmish2006@gmail.com | mohdhamzakhan0101@gmail.com
    </div>
</div>
""",
unsafe_allow_html=True
)