# Lane Web Demo Design

**Goal:** Build a minimal course-project web app for the Lane Segmentation repository that presents the project clearly and supports one real interaction: upload a road image and display the model's lane-segmentation result.

**Scope**
- Single-page web UI only
- Real backend inference for one uploaded image at a time
- CPU-compatible execution on the current machine
- No login, database, history, user accounts, or batch jobs

**Context**
- The current repository is a research/model repository, not a product web app.
- The existing inference code is CLI-oriented and assumes CUDA in several places.
- The latest project description comes from `README.md`, which frames the project as a lane segmentation model based on InSPyReNet.

**Architecture**
- A small Flask app serves the landing page and a `/predict` upload endpoint.
- A dedicated inference service module loads the model once, auto-selects CPU/GPU, runs preprocessing and prediction, and writes result images to a web-accessible static output directory.
- The frontend is a single HTML page with CSS and light JavaScript. It keeps the visual tone aligned with the supplied design file: white canvas, black CTAs, light-gray cards, dark footer.

**User Experience**
- Hero section introduces the project and links to the demo area.
- Project sections summarize background, workflow, and reported metrics from `README.md`.
- Demo section allows selecting an image, previewing it, running real inference, and viewing the generated output.
- The page shows loading and error states so the interaction feels complete even though the feature set is intentionally small.

**Data Flow**
1. User selects an image in the browser.
2. Frontend sends the file to `POST /predict`.
3. Backend stores the upload in a temporary uploads directory.
4. Inference service runs the model and saves a result image in a static results directory.
5. Backend returns JSON containing the public result URL.
6. Frontend displays the result next to the original preview.

**Error Handling**
- Reject missing files and non-image uploads with clear messages.
- Catch inference failures and return a readable server-side error.
- Ensure CPU fallback is used when CUDA is unavailable.

**Testing**
- Flask test client coverage for the homepage and upload endpoint
- Mocked inference service in tests so UI/API behavior is verified without running the full model
- Manual real-world verification by launching the app and running one actual prediction
