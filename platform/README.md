# Protein Inference Platform Scaffold

This scaffold separates product pieces for a Modal + Supabase + frontend architecture.

## Layout

- `frontend/` UI app (local now, Vercel later)
- `api/` backend endpoints for job submission/status
- `workers/modal/` GPU job workers
- `supabase/` database schema and setup notes

## Expected flow

1. Frontend calls API to submit a job.
2. API creates a `jobs` row in Supabase.
3. API triggers Modal worker with `job_id` and input paths.
4. Modal worker writes artifacts and updates `jobs.status`.
5. Frontend subscribes to Supabase realtime updates.
