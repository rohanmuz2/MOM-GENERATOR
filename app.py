from flask import Flask, request, jsonify
import vertexai
from langchain.llms import VertexAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain import PromptTemplate
import time
from typing import List

# Initialize Flask app
app = Flask(__name__)

# Vertex AI and project initialization
PROJECT_ID = "project-technovate"
vertexai.init(project=PROJECT_ID, location="us-central1")

# Rate limiter utility function
def rate_limit(max_per_minute):
    period = 60 / max_per_minute
    while True:
        before = time.time()
        yield
        after = time.time()
        elapsed = after - before
        sleep_time = max(0, period - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

# LLM model configuration
llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1020,
    temperature=0.1,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

# Flask route for summarizing meeting content
@app.route('/summarize_meeting', methods=['POST'])
def summarize_meeting():
    # Extract meeting content from POST request
    meeting_content = request.json['meeting_content']

    # Create a Document object
    doc = Document(
        page_content=meeting_content,
        metadata={
            "source": "Company Meeting Transcript",
            "date": "YYYY-MM-DD",  # Replace with the actual date of the meeting
            "topic": "API Development Discussion"
        }
    )

    # Set up the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    texts = text_splitter.split_documents([doc])

    # Define prompt templates
    prompt_template = """Identify and list the key points, action items, pending issues, and responsible individuals from the following transcript:
                        {text}
                        MEETING MINUTES:
                        - Key Points:
                        - Action Items:
                        - Pending Issues:
                        - Responsible Individuals:
                    """
    refine_template = (
        "Your task is to refine the meeting minutes summary.\n"
        "Below is the existing summary up to a certain point: {existing_answer}\n"
        "You are to enhance the existing summary with additional details from the new context provided.\n"
        "If the new information is not relevant, maintain the original summary.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Refine the meeting minutes to include any missing action items, pending issues, or responsible individuals from the new context. If the new context doesn't add value, return the original summary."
    )

    prompt = PromptTemplate.from_template(prompt_template)
    refine_prompt = PromptTemplate.from_template(refine_template)

    # Load summarization chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    # Execute summarization chain
    result = chain({"input_documents": texts}, return_only_outputs=True)

    #print for debug
    print(result["output_text"])
    # Return the result as JSON
    return jsonify({'data': result["output_text"]})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True,port= 9090)



# docker build -t my-flask-app . -f Docker
# docker run -p 4000:8081 my-flask-app

