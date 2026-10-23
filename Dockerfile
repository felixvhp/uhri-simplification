# Feedback form server for Google Cloud Run (or any container host).
# Only the feedback form ships here: it is stdlib-only, so nothing is pip-installed.
#
#   gcloud run deploy uhri-feedback --source . --region europe-west6
#
# Before building, generate the review set so it is baked into the image:
#   python -m src.feedback_form.make_data
# (or, for a dataset too big to bake, upload it to the bucket and set
#  FEEDBACK_RECORDS_PATH=/data/records/review_data.jsonl on the service)
#
# Submissions are written to $FEEDBACK_EXPORTS_DIR; mount a Cloud Storage
# bucket there (see README "Deploying the feedback form") so they survive
# redeploys. Cloud Run sets $PORT itself.
FROM python:3.11-slim

WORKDIR /app
COPY src/ ./src/

ENV PYTHONUNBUFFERED=1 \
    FEEDBACK_EXPORTS_DIR=/data/exports

EXPOSE 8080
CMD ["python", "-m", "src.feedback_form.serve", "--host", "0.0.0.0"]
