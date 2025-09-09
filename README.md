<artifact artifact_id="56dc7f52-6028-4c3a-8b9e-6bec3036c758" artifact_version_id="1a2b3c4d-5e6f-7890-abcd-ef1234567890" title="README.md" contenttype="text/markdown">

<h2>Overview</h2>
<p>The CV_Ranking module is designed to streamline the recruitment process by automating the selection and ranking of resumes based on their relevance to a provided job description. The system minimizes human effort and time while ensuring the identification of the most suitable candidates through advanced text processing and machine learning techniques.</p>
<h2>🎯 Objectives</h2>
<ul>
<li>Automate the screening and ranking of resumes to reduce manual effort in recruitment.</li>
<li>Accurately match candidate profiles to job requirements using natural language processing (NLP) and machine learning.</li>
<li>Provide recruiters with structured candidate data and a ranked list of resumes for efficient decision-making.</li>
</ul>
<h2>⚙️ Technical Pipeline</h2>
<p>The module follows a robust pipeline to process, analyze, and rank resumes. Below is an overview of each stage:</p>
<h3>1. File and Session Management</h3>
<ul>
<li><strong>Unique Identification</strong>: Each submission is assigned a unique UUID to manage sessions independently.</li>
<li><strong>Temporary Storage</strong>: Resumes are stored in organized directories (<code>uploads</code>, <code>temp</code>, <code>results</code>) for processing.</li>
<li><strong>Data Privacy</strong>: Files are automatically deleted after processing to optimize disk space and ensure confidentiality.</li>
</ul>
<h3>2. Resume Preprocessing</h3>
<ul>
<li><strong>PDF Conversion</strong>: PDF resumes are converted to images per page using the <code>pdf2image</code> library and Poppler.</li>
<li><strong>Image Support</strong>: Direct processing of JPG/JPEG files is supported for OCR integration.</li>
</ul>
<h3>3. Text Line Detection (CRAFT)</h3>
<ul>
<li><strong>Text Localization</strong>: The CRAFT model detects text regions on each page of the resume.</li>
<li><strong>Line Segmentation</strong>: Detected text lines are saved individually to enhance OCR accuracy.</li>
</ul>
<h3>4. Optical Character Recognition (TrOCR)</h3>
<ul>
<li><strong>Text Extraction</strong>: Text lines are processed using the TrOCR model to generate readable paragraphs.</li>
<li><strong>Text Cleaning</strong>: The <code>clean_ocr_text</code> module normalizes text by removing special characters, correcting accents, and standardizing typography.</li>
<li><strong>Multi-Page Handling</strong>: Text from all pages of a resume is concatenated to form a cohesive output.</li>
</ul>
<h3>5. Structured Information Extraction (LLaMA)</h3>
<ul>
<li><strong>Text Processing</strong>: Cleaned text is passed to a LLaMA model (via <code>llama.cpp</code>) for structured data extraction.</li>
<li><strong>Output Format</strong>: Extracted information is saved as a JSON file containing:
<ul>
<li>Name</li>
<li>Email</li>
<li>Work Experience</li>
<li>Skills</li>
<li>Education</li>
</ul>
</li>
<li><strong>Storage</strong>: Structured data is saved per resume for downstream processing.</li>
</ul>
<h3>6. Similarity Scoring</h3>
<ul>
<li><strong>Text Representation</strong>: Resumes and job descriptions are encoded using the <code>SentenceTransformer</code> model (<code>all-MiniLM-L6-v2</code>).</li>
<li><strong>Similarity Calculation</strong>: Cosine similarity is computed between resume and job description vectors, yielding a relevance score between 0 and 1.</li>
<li><strong>Relevance</strong>: Higher scores indicate greater alignment with the job requirements.</li>
</ul>
<h3>7. Resume Ranking</h3>
<ul>
<li><strong>Sorting</strong>: Resumes are ranked in descending order based on their similarity scores.</li>
<li><strong>Output</strong>: Recruiters receive a prioritized list of resumes, each accompanied by structured data and relevance scores.</li>
</ul>
<h2>💻 User Interface &#x26; Results</h2>
<p>The application provides an intuitive interface for submitting job descriptions and resumes. Key features include:</p>
<ul>
<li><strong>Submission Interface</strong>: Users can upload resumes and input job descriptions seamlessly.</li>
<li><strong>Result Visualization</strong>: Displays a ranked list of resumes with:
<ul>
<li>Similarity scores</li>
<li>Extracted structured information (name, email, experience, skills, education)</li>
</ul>
</li>
</ul>
<h3>Example Output</h3>
<ul>
<li><strong>Figure 6.1</strong>: Screenshot of the user interface for resume and job description submission.</li>
<li><strong>Figure 6.2</strong>: Sample output showing the ranked resumes with similarity scores and structured data.</li>
</ul>
<h2>🛠️ Technologies Used</h2>
<ul>
<li><strong>File Processing</strong>: <code>pdf2image</code>, Poppler</li>
<li><strong>Text Detection</strong>: CRAFT model</li>
<li><strong>OCR</strong>: TrOCR</li>
<li><strong>Text Normalization</strong>: Custom <code>clean_ocr_text</code> module</li>
<li><strong>NLP</strong>: LLaMA (via <code>llama.cpp</code>), SentenceTransformer (<code>all-MiniLM-L6-v2</code>)</li>
<li><strong>Session Management</strong>: UUID-based organization</li>
<li><strong>Storage</strong>: Temporary directories (<code>uploads</code>, <code>temp</code>, <code>results</code>)</li>
</ul>
<h2>📋 Requirements</h2>
<ul>
<li>Python 3.8+</li>
<li>Libraries: <code>pdf2image</code>, <code>llama.cpp</code>, <code>sentence-transformers</code>, <code>poppler-utils</code></li>
<li>Models: CRAFT, TrOCR, LLaMA, SentenceTransformer (<code>all-MiniLM-L6-v2</code>)</li>
<li>Disk space for temporary file storage</li>
<li>Poppler installed for PDF-to-image conversion</li>
</ul>
<h2>🚀 Getting Started</h2>
<ol>
<li><strong>Clone the Repository</strong>:
<pre><code class="language-bash">git clone &#x3C;repository_url>
</code></pre>
</li>
<li><strong>Install Dependencies</strong>:
<pre><code class="language-bash">pip install -r requirements.txt
</code></pre>
</li>
<li><strong>Configure Poppler</strong>: Ensure Poppler is installed and added to the system PATH.</li>
<li><strong>Run the Application</strong>:
<pre><code class="language-bash">python main.py
</code></pre>
</li>
<li><strong>Access the Interface</strong>: Open the provided URL in a browser to upload resumes and job descriptions.</li>
</ol>
<h2>🔐 Data Privacy</h2>
<ul>
<li>All uploaded files are stored temporarily and deleted post-processing.</li>
<li>UUID-based session management ensures secure and independent handling of submissions.</li>
</ul>
<h2>📜 License</h2>
<p>This project is licensed under the MIT License. See the <code>LICENSE</code> file for details.</p>
<h2>📬 Contact</h2>
<p>For inquiries or contributions, please contact the project maintainers at [<a href="mailto:your-email@example.com">your-email@example.com</a>].
</p></artifact>
