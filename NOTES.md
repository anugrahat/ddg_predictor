Learning Notes (Modal + Supabase + Frontend)

-- Goal: build a Tamarind/Neurosnap-style protein inference app.

-- Stack choice
-- Modal = GPU compute + job execution (ProteinMPNN / ESMFold / Protenix)
-- Supabase = auth + Postgres + storage + realtime job status
-- Frontend now = local development in Codex
-- Frontend later = Vercel deployment

-- What each part does
-- Modal: run inference jobs, autoscale workers, stream logs, return status
-- Supabase: users, job records, input/output metadata, signed URLs, live status updates
-- Frontend: submit job, track status, view/download results

-- Core job flow
-- 1. User submits job in UI
-- 2. API creates jobs row in Supabase (status = queued)
-- 3. API triggers Modal with job_id + input artifact paths
-- 4. Modal runs model, writes outputs, updates Supabase status (running/completed/failed)
-- 5. UI subscribes to Supabase realtime and renders status/results

-- Build order (MVP)
-- 1. Create Supabase project + schema (jobs, artifacts)
-- 2. Build one Modal worker first (ProteinMPNN)
-- 3. Add submit-job API endpoint
-- 4. Build basic frontend locally
-- 5. Deploy frontend to Vercel later
