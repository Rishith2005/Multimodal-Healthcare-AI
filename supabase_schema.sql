-- ============================================================================
-- Supabase Table: patient_records
-- For the Multimodal Healthcare Diagnostic System
--
-- HOW TO USE:
--   1. Go to your Supabase project dashboard
--   2. Navigate to SQL Editor (left sidebar)
--   3. Paste this entire SQL and click "Run"
-- ============================================================================

CREATE TABLE IF NOT EXISTS patient_records (
  id            UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  created_at    TIMESTAMPTZ DEFAULT now() NOT NULL,

  -- Core patient info
  patient_name      TEXT NOT NULL,
  gender            TEXT NOT NULL,
  age               NUMERIC NOT NULL,
  height            NUMERIC NOT NULL,
  weight            NUMERIC NOT NULL,
  bmi               NUMERIC,
  blood_glucose     NUMERIC NOT NULL,
  blood_pressure    NUMERIC NOT NULL,
  family_history    TEXT NOT NULL,
  physical_activity TEXT NOT NULL,
  smoking_status    TEXT NOT NULL,
  medical_conditions TEXT NOT NULL,
  symptoms          TEXT[] DEFAULT '{}',
  hba1c             NUMERIC NOT NULL,
  insulin_level     NUMERIC NOT NULL,

  -- Optional advanced fields
  pregnancies         NUMERIC,
  skin_thickness      NUMERIC,
  diabetes_pedigree   NUMERIC,
  cholesterol         NUMERIC,
  heart_rate          NUMERIC,
  oxygen_saturation   NUMERIC,

  -- Prediction results
  prediction                TEXT,
  confidence                NUMERIC,
  clinical_interpretation   TEXT
);

-- Index for fast lookups by date and patient name
CREATE INDEX IF NOT EXISTS idx_patient_records_created_at
  ON patient_records (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_patient_records_name
  ON patient_records (patient_name);

-- Enable Row Level Security (RLS) — adjust policies as needed
ALTER TABLE patient_records ENABLE ROW LEVEL SECURITY;

-- Allow anonymous reads/writes for the prototype (tighten for production)
CREATE POLICY "Allow anonymous insert"
  ON patient_records FOR INSERT
  WITH CHECK (true);

CREATE POLICY "Allow anonymous select"
  ON patient_records FOR SELECT
  USING (true);
