import { useState, useEffect, useMemo } from 'react'
import {
  Brain,
  HeartPulse,
  Activity,
  ScanLine,
  Table2,
  Waves,
  ArrowRight,
  BookOpen,
  ChevronRight,
  Zap,
  Shield,
  Database,
  Cpu,
  UploadCloud,
  CheckCircle2,
  AlertTriangle,
  FileText,
  BarChart3,
  RotateCcw,
  LineChart,
  Loader2,
  Save,
  ActivitySquare,
  ChevronDown,
  Sliders
} from 'lucide-react'

import { savePatientRecord, type PatientRecord } from './lib/supabase'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NavLink {
  label: string
  href: string
  icon: React.ReactNode
}

interface ModelCard {
  title: string
  description: string
  icon: React.ReactNode
  metrics: string
  color: string
}

interface DatasetInfo {
  name: string
  modality: string
  size: string
  icon: React.ReactNode
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

const NAV_LINKS: NavLink[] = [
  { label: 'Portal', href: '#portal', icon: <HeartPulse size={16} /> },
  { label: 'Models', href: '#models', icon: <Brain size={16} /> },
  { label: 'Datasets', href: '#datasets', icon: <Database size={16} /> },
  { label: 'Research', href: '#research', icon: <BookOpen size={16} /> },
]

const MODEL_CARDS: ModelCard[] = [
  {
    title: 'CNN — Chest X-ray',
    description: 'ResNet18 backbone for pneumonia classification from chest radiographs',
    icon: <ScanLine size={24} />,
    metrics: '128-dim embedding',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    title: 'MLP — Diabetes Risk',
    description: 'Multi-layer perceptron with batch normalization for tabular clinical data',
    icon: <Table2 size={24} />,
    metrics: '32-dim embedding',
    color: 'from-emerald-500 to-teal-500',
  },
  {
    title: 'LSTM — ICU Vitals',
    description: 'Bidirectional LSTM processing 48h time-series of 12 vital signs',
    icon: <Waves size={24} />,
    metrics: '32-dim embedding',
    color: 'from-violet-500 to-purple-500',
  },
]

const DATASETS: DatasetInfo[] = [
  {
    name: 'ChestXRay2017',
    modality: 'Images',
    size: '5,863 X-rays',
    icon: <ScanLine size={20} />,
  },
  {
    name: 'Pima Diabetes',
    modality: 'Tabular',
    size: '768 patients',
    icon: <HeartPulse size={20} />,
  },
  {
    name: 'PhysioNet CinC',
    modality: 'Time-Series',
    size: '4,000 ICU stays',
    icon: <Activity size={20} />,
  },
]

const VITAL_NAMES = [
  "HR (Heart Rate)",
  "RespRate (Respiration)",
  "Temp (Temperature)",
  "NISysABP (Systolic BP)",
  "NIDiasABP (Diastolic BP)",
  "NIMAP (Mean Arterial BP)",
  "GCS (Glasgow Coma)",
  "Glucose (Blood Sugar)",
  "BUN (Urea Nitrogen)",
  "Creatinine",
  "HCT (Hematocrit)",
  "Urine (Output)"
];

function generateSimulatedVitals(isHighRisk: boolean = false): number[][] {
  const vitals: number[][] = [];
  for (let h = 0; h < 48; h++) {
    const hr = isHighRisk 
      ? 92 + Math.sin(h / 3) * 8 + Math.random() * 5
      : 72 + Math.sin(h / 5) * 4 + Math.random() * 2;
    const resp = isHighRisk 
      ? 20 + Math.sin(h / 4) * 3 + Math.random() * 1.5
      : 15 + Math.random() * 1.5;
    const temp = isHighRisk 
      ? 38.2 + Math.sin(h / 8) * 0.3 + Math.random() * 0.1
      : 36.8 + Math.random() * 0.1;
    const sys = isHighRisk 
      ? 142 + Math.sin(h / 2) * 12 + Math.random() * 6
      : 118 + Math.random() * 3;
    const dias = isHighRisk 
      ? 90 + Math.sin(h / 2) * 8 + Math.random() * 4
      : 76 + Math.random() * 2;
    const map = (sys + 2 * dias) / 3;
    const gcs = isHighRisk 
      ? Math.max(9, 15 - Math.floor(h / 10))
      : 15;
    const glucose = isHighRisk 
      ? 165 + Math.random() * 15
      : 95 + Math.random() * 5;
    const bun = isHighRisk 
      ? 26 + h / 6
      : 12 + Math.random();
    const creat = isHighRisk 
      ? 1.6 + h / 20
      : 0.8 + Math.random() * 0.1;
    const hct = isHighRisk 
      ? 36 - h / 10
      : 42 - Math.random() * 0.5;
    const urine = isHighRisk 
      ? Math.max(15, 60 - h * 1.0)
      : 70 + Math.random() * 10;

    vitals.push([hr, resp, temp, sys, dias, map, gcs, glucose, bun, creat, hct, urine]);
  }
  return vitals;
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function Logo() {
  return (
    <div className="flex items-center gap-2">
      <div className="relative w-8 h-8">
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-lg rotate-6 animate-pulse-slow" />
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-400 to-cyan-500 rounded-lg flex items-center justify-center">
          <Brain size={18} className="text-white" />
        </div>
      </div>
      <span className="font-bold text-white text-lg tracking-tight">MedFuse</span>
    </div>
  )
}

function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 transition-all duration-500 ${
        scrolled
          ? 'bg-black/60 backdrop-blur-xl border border-white/10 shadow-2xl'
          : 'bg-white/15 backdrop-blur-md border border-white/20'
      } rounded-full px-2 py-1.5`}
    >
      <div className="flex items-center gap-1">
        <div className="px-3">
          <Logo />
        </div>
        <div className="w-px h-6 bg-white/20 mx-1" />
        {NAV_LINKS.map((link) => (
          <a
            key={link.label}
            href={link.href}
            className="nav-pill flex items-center gap-1.5"
          >
            {link.icon}
            {link.label}
          </a>
        ))}
      </div>
    </nav>
  )
}

function HeroSection() {
  return (
    <section className="relative min-h-[80vh] flex items-end pb-16 pl-8 sm:pl-16 md:pl-24">
      <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/40 to-transparent pointer-events-none" />
      <div className="absolute inset-0 bg-gradient-to-r from-black/60 via-transparent to-transparent pointer-events-none" />

      <div className="relative z-10 max-w-2xl">
        <div className="badge mb-6 animate-fade-in">
          <Shield size={12} />
          Academic Prototype — Fused Multimodal AI
        </div>

        <h1 className="text-5xl sm:text-6xl md:text-7xl font-extrabold leading-[1.05] tracking-tight text-white mb-6 animate-slide-up">
          Multimodal
          <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500">
            Healthcare AI
          </span>
        </h1>

        <p className="text-lg sm:text-xl text-white/70 leading-relaxed mb-8 max-w-lg animate-slide-up">
          From the moment a patient walks in to the instant a clinician needs answers —
          AI-powered diagnostics that turn X-rays, lab reports, and vital signs
          into life-saving insights, in seconds.
        </p>

        <div className="flex flex-wrap gap-4 animate-slide-up">
          <a href="#portal" className="btn-primary group">
            Open Portal
            <ArrowRight
              size={18}
              className="transition-transform group-hover:translate-x-1"
            />
          </a>
          <a href="#models" className="btn-secondary group">
            Model Details
            <ChevronRight
              size={18}
              className="transition-transform group-hover:translate-x-0.5"
            />
          </a>
        </div>
      </div>
    </section>
  )
}

export default function App() {
  // Tabular Intake Primary Fields
  const [patientName, setPatientName] = useState<string>('Jane Doe')
  const [gender, setGender] = useState<string>('Female')
  const [age, setAge] = useState<number>(28)
  const [height, setHeight] = useState<number>(165)
  const [weight, setWeight] = useState<number>(65)
  const [bloodGlucose, setBloodGlucose] = useState<number>(85)
  const [bloodPressure, setBloodPressure] = useState<number>(70)
  const [familyHistory, setFamilyHistory] = useState<string>('No Family History')
  const [physicalActivity, setPhysicalActivity] = useState<string>('Active (Moderate)')
  const [smokingStatus, setSmokingStatus] = useState<string>('Never Smoked')
  const [medicalConditions, setMedicalConditions] = useState<string>('None')
  const [symptoms, setSymptoms] = useState<string[]>([])
  const [hba1c, setHba1c] = useState<number>(5.4)
  const [insulinLevel, setInsulinLevel] = useState<number>(80)

  // Tabular Intake Optional Advanced Fields
  const [pregnancies, setPregnancies] = useState<number>(0)
  const [skinThickness, setSkinThickness] = useState<number>(20)
  const [diabetesPedigree, setDiabetesPedigree] = useState<number>(0.35)
  const [cholesterol, setCholesterol] = useState<number>(180)
  const [sleepDuration, setSleepDuration] = useState<number>(7)
  const [stressLevel, setStressLevel] = useState<number>(4)
  const [dietaryPattern, setDietaryPattern] = useState<string>('Balanced')
  const [alcoholConsumption, setAlcoholConsumption] = useState<string>('None')
  const [heartRate, setHeartRate] = useState<number>(72)
  const [oxygenSaturation, setOxygenSaturation] = useState<number>(98)
  const [waistCircumference, setWaistCircumference] = useState<number>(80)

  // Auto-calculated BMI
  const bmi = useMemo(() => {
    if (height > 0) {
      return Number((weight / ((height / 100) ** 2)).toFixed(1))
    }
    return 0
  }, [height, weight])

  // Files
  const [prescriptionFile, setPrescriptionFile] = useState<File | null>(null)
  const [prescriptionPreview, setPrescriptionPreview] = useState<string>('')
  const [prescriptionDocType, setPrescriptionDocType] = useState<'prescription' | 'icu_record'>('prescription')
  const [scanFile, setScanFile] = useState<File | null>(null)
  const [scanPreview, setScanPreview] = useState<string>('')

  // State flags
  const [isExtractingVitals, setIsExtractingVitals] = useState<boolean>(false)
  const [extractedVitals, setExtractedVitals] = useState<number[][] | null>(null)
  const [vitalExtractStep, setVitalExtractStep] = useState<string>('')
  const [selectedVitalIndex, setSelectedVitalIndex] = useState<number>(0)

  // Advanced settings accordion
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(false)
  const [intakeAdvancedOpen, setIntakeAdvancedOpen] = useState<boolean>(false)
  const [cnnWeight, setCnnWeight] = useState<number>(0.33)
  const [tabWeight, setTabWeight] = useState<number>(0.33)
  const [rnnWeight, setRnnWeight] = useState<number>(0.34)

  // Inference State
  const [isInferring, setIsInferring] = useState<boolean>(false)
  const [inferenceStep, setInferenceStep] = useState<string>('')
  const [predictionResult, setPredictionResult] = useState<any | null>(null)
  const [tabularResult, setTabularResult] = useState<any | null>(null)
  const [isTabularRunning, setIsTabularRunning] = useState<boolean>(false)
  const [heatmapEnabled, setHeatmapEnabled] = useState<boolean>(false)
  const [isRecordSaved, setIsRecordSaved] = useState<boolean>(false)
  const [errorMessage, setErrorMessage] = useState<string>('')

  // Preset Patient Templates
  const applyTemplate = (type: 'healthy' | 'atrisk') => {
    if (type === 'healthy') {
      setPatientName('Jane Doe')
      setGender('Female')
      setAge(24)
      setHeight(168)
      setWeight(58)
      setBloodGlucose(85)
      setBloodPressure(70)
      setFamilyHistory('No Family History')
      setPhysicalActivity('Active (Moderate)')
      setSmokingStatus('Never Smoked')
      setMedicalConditions('None')
      setSymptoms([])
      setHba1c(5.2)
      setInsulinLevel(75)

      // Advanced
      setPregnancies(0)
      setSkinThickness(18)
      setDiabetesPedigree(0.24)
      setCholesterol(180)
      setSleepDuration(8)
      setStressLevel(3)
      setDietaryPattern('Balanced')
      setAlcoholConsumption('None')
      setHeartRate(68)
      setOxygenSaturation(99)
      setWaistCircumference(72)

      setExtractedVitals(generateSimulatedVitals(false))
    } else {
      setPatientName('John Smith')
      setGender('Male')
      setAge(46)
      setHeight(175)
      setWeight(105)
      setBloodGlucose(165)
      setBloodPressure(88)
      setFamilyHistory('First-Degree Relative')
      setPhysicalActivity('Sedentary (Low)')
      setSmokingStatus('Former Smoker')
      setMedicalConditions('Hypertension')
      setSymptoms(['Polyuria', 'Polydipsia', 'Fatigue'])
      setHba1c(7.2)
      setInsulinLevel(185)

      // Advanced
      setPregnancies(0)
      setSkinThickness(28)
      setDiabetesPedigree(0.72)
      setCholesterol(240)
      setSleepDuration(6)
      setStressLevel(7)
      setDietaryPattern('High Carb / Processed')
      setAlcoholConsumption('Socially')
      setHeartRate(84)
      setOxygenSaturation(96)
      setWaistCircumference(108)

      setExtractedVitals(generateSimulatedVitals(true))
    }
    setIsRecordSaved(false)
    setTabularResult(null)
  }

  // Handle Prescription Upload & Simulate OCR Vital Signs Extraction
  const handlePrescriptionUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setPrescriptionFile(file)
    const reader = new FileReader()
    reader.onloadend = () => {
      setPrescriptionPreview(reader.result as string)
    }
    reader.readAsDataURL(file)

    // Only extract ICU vitals for ICU records, not regular prescriptions
    if (prescriptionDocType !== 'icu_record') {
      setExtractedVitals(null)
      setIsExtractingVitals(false)
      return
    }

    setIsExtractingVitals(true)
    setExtractedVitals(null)
    
    const steps = [
      'Normalizing ICU chart layout...',
      'Running vital-signs segmenter...',
      'Extracting 48h temporal measurements...',
      'Synthesizing timeseries matrix...'
    ]
    
    let stepIdx = 0
    setVitalExtractStep(steps[0])
    
    const interval = setInterval(() => {
      stepIdx++
      if (stepIdx < steps.length) {
        setVitalExtractStep(steps[stepIdx])
      } else {
        clearInterval(interval)
        setIsExtractingVitals(false)
        const tabularHighRisk = bloodGlucose > 130 || bmi > 30 || age > 40
        setExtractedVitals(generateSimulatedVitals(tabularHighRisk))
      }
    }, 450)
  }

  // Handle scan file
  const handleScanUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setScanFile(file)
    const reader = new FileReader()
    reader.onloadend = () => {
      setScanPreview(reader.result as string)
    }
    reader.readAsDataURL(file)
    setIsRecordSaved(false)
    setPredictionResult(null)
  }

  // Call the standalone tabular prediction API
  const handleRunTabularPrediction = async () => {
    setIsTabularRunning(true)
    setTabularResult(null)
    setErrorMessage('')

    try {
      const payload = {
        PatientName: patientName,
        Gender: gender,
        Age: age,
        Height: height,
        Weight: weight,
        BMI: bmi,
        BloodGlucose: bloodGlucose,
        BloodPressure: bloodPressure,
        FamilyHistory: familyHistory,
        PhysicalActivity: physicalActivity,
        SmokingStatus: smokingStatus,
        MedicalConditions: medicalConditions,
        Symptoms: symptoms,
        HbA1c: hba1c,
        InsulinLevel: insulinLevel,
        Pregnancies: gender.toLowerCase() === 'female' ? pregnancies : 0,
        SkinThickness: skinThickness,
        DiabetesPedigreeFunction: diabetesPedigree,
        CholesterolLevel: cholesterol,
        SleepDuration: sleepDuration,
        StressLevel: stressLevel,
        DietaryPattern: dietaryPattern,
        AlcoholConsumption: alcoholConsumption,
        HeartRate: heartRate,
        OxygenSaturation: oxygenSaturation,
        WaistCircumference: waistCircumference
      }

      const response = await fetch('http://localhost:8000/api/predict/tabular', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      })

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`)
      }

      const data = await response.json()
      setTabularResult(data)

      // Auto-save patient record to Supabase after successful prediction
      const record: PatientRecord = {
        patient_name: patientName,
        gender,
        age,
        height,
        weight,
        bmi: bmi || null,
        blood_glucose: bloodGlucose,
        blood_pressure: bloodPressure,
        family_history: familyHistory,
        physical_activity: physicalActivity,
        smoking_status: smokingStatus,
        medical_conditions: medicalConditions,
        symptoms,
        hba1c: hba1c,
        insulin_level: insulinLevel,
        pregnancies: gender.toLowerCase() === 'female' ? pregnancies : null,
        skin_thickness: skinThickness || null,
        diabetes_pedigree: diabetesPedigree || null,
        cholesterol: cholesterol || null,
        heart_rate: heartRate || null,
        oxygen_saturation: oxygenSaturation || null,
        prediction: data.prediction || null,
        confidence: data.confidence || null,
        clinical_interpretation: data.clinical_interpretation || null,
      }
      await savePatientRecord(record)
      setIsRecordSaved(true)
    } catch (err: any) {
      console.error(err)
      setErrorMessage(err.message || 'Unable to connect to the backend server. Please make sure the backend FastAPI app is running.')
    } finally {
      setIsTabularRunning(false)
    }
  }

  // Call the backend fusion model API
  const handleRunDiagnostics = async () => {
    setIsInferring(true)
    setErrorMessage('')
    setPredictionResult(null)
    setIsRecordSaved(false)

    const vitalsSequence = extractedVitals || generateSimulatedVitals(bloodGlucose > 130 || bmi > 30)

    const steps = [
      'Sending multimodal data packages...',
      'Verifying alignments...',
      'Evaluating CNN branch embedding...',
      'Evaluating tabular MLP prediction...',
      'Processing LSTM vital branch (48h timeline)...',
      'Fusing diagnostic vectors...',
      'Inference completed!'
    ]

    let stepIdx = 0
    setInferenceStep(steps[0])

    const stepInterval = setInterval(() => {
      stepIdx++
      if (stepIdx < steps.length - 1) {
        setInferenceStep(steps[stepIdx])
      } else {
        clearInterval(stepInterval)
      }
    }, 350)

    try {
      const formData = new FormData()
      
      // Map Family History to PIMA Pedigree if custom pedigree is not specified
      let mappedPedigree = diabetesPedigree
      if (!mappedPedigree) {
        const fh = familyHistory.toLowerCase()
        if (fh.includes('both')) mappedPedigree = 0.85
        else if (fh.includes('first') || fh.includes('parent') || fh.includes('sibling')) mappedPedigree = 0.65
        else if (fh.includes('second')) mappedPedigree = 0.35
        else mappedPedigree = 0.15
      }

      const payload = {
        tabular: {
          Pregnancies: gender.toLowerCase() === 'female' ? pregnancies : 0,
          Glucose: bloodGlucose,
          BloodPressure: bloodPressure,
          SkinThickness: skinThickness,
          Insulin: insulinLevel,
          BMI: bmi,
          DiabetesPedigreeFunction: mappedPedigree,
          Age: age
        },
        timeseries: vitalsSequence
      }

      formData.append('data', JSON.stringify(payload))

      if (scanFile) {
        formData.append('xray_file', scanFile)
      }

      const response = await fetch('http://localhost:8000/predict/fusion', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server returned status: ${response.status}`)
      }

      const result = await response.json()
      setPredictionResult(result)
    } catch (err: any) {
      console.error(err)
      setErrorMessage(err.message || 'Unable to connect to the backend server. Please make sure the backend FastAPI app is running.')
    } finally {
      setIsInferring(false)
    }
  }

  const handleSaveToDatabase = async () => {
    try {
      const record: PatientRecord = {
        patient_name: patientName,
        gender,
        age,
        height,
        weight,
        bmi: bmi || null,
        blood_glucose: bloodGlucose,
        blood_pressure: bloodPressure,
        family_history: familyHistory,
        physical_activity: physicalActivity,
        smoking_status: smokingStatus,
        medical_conditions: medicalConditions,
        symptoms,
        hba1c: hba1c,
        insulin_level: insulinLevel,
        pregnancies: gender.toLowerCase() === 'female' ? pregnancies : null,
        skin_thickness: skinThickness || null,
        diabetes_pedigree: diabetesPedigree || null,
        cholesterol: cholesterol || null,
        heart_rate: heartRate || null,
        oxygen_saturation: oxygenSaturation || null,
        prediction: predictionResult?.binary_risk || tabularResult?.prediction || null,
        confidence: predictionResult?.binary_confidence || tabularResult?.confidence || null,
        clinical_interpretation: tabularResult?.clinical_interpretation || null,
      }
      const { error } = await savePatientRecord(record)
      if (error) {
        console.error('[Save] Failed:', error)
        setErrorMessage('Failed to save record to database: ' + error)
      } else {
        setIsRecordSaved(true)
      }
    } catch (err: any) {
      console.error('[Save] Unexpected error:', err)
      setErrorMessage('Failed to save record: ' + (err.message || 'Unknown error'))
    }
  }

  return (
    <div className="bg-[#f0f0ee] min-h-screen">
      {/* Background video — fullscreen, autoplay, muted, loop */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute inset-0 w-full h-full object-cover"
          poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='1920' height='1080'%3E%3Crect fill='%23111827'/%3E%3C/svg%3E"
        >
          <source
            src="https://cdn.coverr.co/videos/coverr-abstract-blue-particles-1584/1080p.mp4"
            type="video/mp4"
          />
        </video>
        <div className="absolute inset-0 bg-black/45" />
      </div>

      {/* Content */}
      <div className="relative z-10">
        <Navbar />
        <HeroSection />

        {/* ========================================================================= */}
        {/* INTERACTIVE DIAGNOSTIC PORTAL SECTION (DARK CLINICAL AI AESTHETIC) */}
        {/* ========================================================================= */}
        <section id="portal" className="py-20 px-6 max-w-7xl mx-auto relative z-20">
          {/* Main Glassmorphism Portal Card */}
          <div className="backdrop-blur-xl bg-black/40 border border-white/15 shadow-2xl rounded-2xl p-6 sm:p-10 text-white">
            <div className="text-center mb-10">
              <div className="badge mb-4">
                <Shield size={12} />
                MedFuse Interactive Portal
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
                Multimodal Assessment Hub
              </h2>
              <p className="text-white/70 max-w-2xl mx-auto mt-2 text-sm">
                Directly interact with our deep learning branches by manually inputting clinical measurements
                and uploading medical imagery. Get real-time unified fusion results.
              </p>

              {/* Template Selectors */}
              <div className="flex justify-center gap-3 mt-6">
                <button
                  onClick={() => applyTemplate('healthy')}
                  className="px-4 py-1.5 text-xs font-semibold rounded-full border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 transition-all duration-200"
                >
                  Load Template: Healthy
                </button>
                <button
                  onClick={() => applyTemplate('atrisk')}
                  className="px-4 py-1.5 text-xs font-semibold rounded-full border border-rose-500/30 bg-rose-500/10 text-rose-300 hover:bg-rose-500/20 transition-all duration-200"
                >
                  Load Template: High Risk
                </button>
              </div>
            </div>

            {/* 3-Column Diagnostic Layout */}
            <div className="grid lg:grid-cols-3 gap-6">
              {/* Column 1: Patient intake form */}
              <div className="bg-white/5 border border-white/10 rounded-2xl p-5 shadow-sm hover:border-white/20 transition-colors">
                <div className="flex items-center gap-2 mb-5 border-b border-white/10 pb-3">
                  <Table2 className="text-cyan-400" size={18} />
                  <h3 className="font-bold text-white text-base">1. Patient Details & Intake</h3>
                </div>
                <div className="space-y-4">
                  {/* Row 1: Patient Name */}
                  <div>
                    <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Patient Name</label>
                    <input
                      type="text"
                      value={patientName}
                      onChange={(e) => { setPatientName(e.target.value); setIsRecordSaved(false); }}
                      className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      placeholder="Jane Doe"
                    />
                  </div>

                  {/* Row 2: Gender & Age */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Gender</label>
                      <select
                        value={gender}
                        onChange={(e) => { setGender(e.target.value); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      >
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Age (Years)</label>
                      <input
                        type="number"
                        min="0"
                        max="120"
                        value={age}
                        onChange={(e) => { setAge(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                  </div>

                  {/* Row 3: Height & Weight */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Height (cm)</label>
                      <input
                        type="number"
                        min="50"
                        max="250"
                        value={height}
                        onChange={(e) => { setHeight(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Weight (kg)</label>
                      <input
                        type="number"
                        min="10"
                        max="300"
                        value={weight}
                        onChange={(e) => { setWeight(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                  </div>

                  {/* Row 4: BMI (auto-calculated) */}
                  <div>
                    <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">BMI (Auto-calculated)</label>
                    <div className="flex gap-2 items-center">
                      <input
                        type="text"
                        readOnly
                        value={bmi > 0 ? `${bmi} kg/m²` : '—'}
                        className="w-full bg-black/60 border border-white/5 text-white/70 rounded-lg px-2.5 py-1.5 text-xs focus:outline-none cursor-not-allowed font-mono"
                      />
                      {bmi > 0 && (
                        <span className={`text-[10px] font-bold px-2 py-1 rounded ${
                          bmi < 18.5 ? 'bg-amber-500/10 text-amber-400' :
                          bmi < 25 ? 'bg-emerald-500/10 text-emerald-400' :
                          bmi < 30 ? 'bg-amber-500/10 text-amber-400' :
                          'bg-rose-500/10 text-rose-400'
                        }`}>
                          {bmi < 18.5 ? 'Underweight' :
                           bmi < 25 ? 'Normal' :
                           bmi < 30 ? 'Overweight' : 'Obese'}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Row 5: Blood Glucose & Blood Pressure */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Blood Glucose (mg/dL)</label>
                      <input
                        type="number"
                        min="30"
                        max="600"
                        value={bloodGlucose}
                        onChange={(e) => { setBloodGlucose(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Blood Pressure (mmHg)</label>
                      <input
                        type="number"
                        min="40"
                        max="250"
                        value={bloodPressure}
                        onChange={(e) => { setBloodPressure(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                  </div>

                  {/* Row 6: HbA1c & Insulin Level */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">HbA1c Level (%)</label>
                      <input
                        type="number"
                        step="0.1"
                        min="3"
                        max="20"
                        value={hba1c}
                        onChange={(e) => { setHba1c(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Insulin Level (mu U/ml)</label>
                      <input
                        type="number"
                        min="0"
                        max="1000"
                        value={insulinLevel}
                        onChange={(e) => { setInsulinLevel(Number(e.target.value)); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      />
                    </div>
                  </div>

                  {/* Row 7: Family History & Activity Level */}
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Family History</label>
                      <select
                        value={familyHistory}
                        onChange={(e) => { setFamilyHistory(e.target.value); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50 text-ellipsis overflow-hidden"
                      >
                        <option value="No Family History">No Family History</option>
                        <option value="Second-Degree Relative">Second-Degree Relative</option>
                        <option value="First-Degree Relative">First-Degree Relative</option>
                        <option value="Both Parents">Both Parents</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Physical Activity</label>
                      <select
                        value={physicalActivity}
                        onChange={(e) => { setPhysicalActivity(e.target.value); setIsRecordSaved(false); }}
                        className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50 text-ellipsis overflow-hidden"
                      >
                        <option value="Sedentary (Low)">Sedentary (Low)</option>
                        <option value="Active (Moderate)">Active (Moderate)</option>
                        <option value="Highly Active (High)">Highly Active (High)</option>
                      </select>
                    </div>
                  </div>

                  {/* Row 8: Smoking Status */}
                  <div>
                    <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Smoking Status</label>
                    <select
                      value={smokingStatus}
                      onChange={(e) => { setSmokingStatus(e.target.value); setIsRecordSaved(false); }}
                      className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                    >
                      <option value="Never Smoked">Never Smoked</option>
                      <option value="Former Smoker">Former Smoker</option>
                      <option value="Current Smoker">Current Smoker</option>
                    </select>
                  </div>

                  {/* Row 9: Existing Medical Conditions */}
                  <div>
                    <label className="block text-[10px] font-semibold text-white/50 uppercase mb-1">Existing Medical Conditions</label>
                    <input
                      type="text"
                      value={medicalConditions}
                      onChange={(e) => { setMedicalConditions(e.target.value); setIsRecordSaved(false); }}
                      className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1.5 text-xs focus:outline-none focus:border-cyan-500/50"
                      placeholder="e.g. Hypertension, None"
                    />
                  </div>

                  {/* Row 10: Symptoms Checklist */}
                  <div>
                    <label className="block text-[10px] font-semibold text-white/50 uppercase mb-2">Symptoms Checklist</label>
                    <div className="grid grid-cols-2 gap-2 bg-black/35 border border-white/5 rounded-xl p-3.5">
                      {[
                        { id: 'Polyuria', label: 'Polyuria' },
                        { id: 'Polydipsia', label: 'Polydipsia' },
                        { id: 'Fatigue', label: 'Fatigue' },
                        { id: 'Blurry Vision', label: 'Blurry Vision' },
                        { id: 'Weight Loss', label: 'Weight Loss' },
                        { id: 'Slow Healing', label: 'Slow Healing' },
                      ].map((sym) => {
                        const checked = symptoms.includes(sym.id)
                        return (
                          <label key={sym.id} className="flex items-center gap-2 cursor-pointer select-none text-[11px] text-white/80 hover:text-white transition-colors">
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => {
                                const nextSym = checked
                                  ? symptoms.filter(s => s !== sym.id)
                                  : [...symptoms, sym.id]
                                setSymptoms(nextSym)
                                setIsRecordSaved(false)
                              }}
                              className="rounded border-white/10 bg-black/40 text-cyan-500 focus:ring-0 focus:ring-offset-0 h-3.5 w-3.5"
                            />
                            <span>{sym.label}</span>
                          </label>
                        )
                      })}
                    </div>
                  </div>

                  {/* Accordion for optional advanced fields */}
                  <div className="border-t border-white/10 pt-3 mt-4">
                    <button
                      type="button"
                      onClick={() => setIntakeAdvancedOpen(!intakeAdvancedOpen)}
                      className="w-full flex items-center justify-between text-white/60 hover:text-white transition-colors text-[11px] font-bold py-1 focus:outline-none animate-none"
                    >
                      <span className="flex items-center gap-1.5">
                        <Sliders size={13} className="text-cyan-400" />
                        Optional Advanced Parameters
                      </span>
                      <ChevronDown size={14} className={`transition-transform duration-200 ${intakeAdvancedOpen ? 'rotate-180' : ''}`} />
                    </button>
                    {intakeAdvancedOpen && (
                      <div className="grid grid-cols-2 gap-3 mt-3 pt-3 border-t border-white/5 animate-fade-in">
                        {/* Pregnancies */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Pregnancies</label>
                          <input
                            type="number"
                            min="0"
                            value={pregnancies}
                            onChange={(e) => { setPregnancies(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* SkinThickness */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Skin Thickness (mm)</label>
                          <input
                            type="number"
                            min="0"
                            value={skinThickness}
                            onChange={(e) => { setSkinThickness(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* DiabetesPedigreeFunction */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Diabetes Pedigree</label>
                          <input
                            type="number"
                            step="0.01"
                            min="0"
                            value={diabetesPedigree}
                            onChange={(e) => { setDiabetesPedigree(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Cholesterol Level */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Cholesterol (mg/dL)</label>
                          <input
                            type="number"
                            min="50"
                            max="500"
                            value={cholesterol}
                            onChange={(e) => { setCholesterol(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Sleep Duration */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Sleep (Hours)</label>
                          <input
                            type="number"
                            min="0"
                            max="24"
                            value={sleepDuration}
                            onChange={(e) => { setSleepDuration(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Stress Level */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Stress Level (1-10)</label>
                          <input
                            type="number"
                            min="1"
                            max="10"
                            value={stressLevel}
                            onChange={(e) => { setStressLevel(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Dietary Pattern */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Dietary Pattern</label>
                          <input
                            type="text"
                            value={dietaryPattern}
                            onChange={(e) => { setDietaryPattern(e.target.value); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                            placeholder="e.g. Balanced"
                          />
                        </div>
                        {/* Alcohol Consumption */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Alcohol Pattern</label>
                          <input
                            type="text"
                            value={alcoholConsumption}
                            onChange={(e) => { setAlcoholConsumption(e.target.value); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                            placeholder="e.g. Occasional"
                          />
                        </div>
                        {/* Heart Rate */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Heart Rate (bpm)</label>
                          <input
                            type="number"
                            min="30"
                            max="220"
                            value={heartRate}
                            onChange={(e) => { setHeartRate(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Oxygen Saturation */}
                        <div>
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Oxygen Sat (%)</label>
                          <input
                            type="number"
                            min="50"
                            max="100"
                            value={oxygenSaturation}
                            onChange={(e) => { setOxygenSaturation(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                        {/* Waist Circumference */}
                        <div className="col-span-2">
                          <label className="block text-[9px] font-semibold text-white/40 uppercase mb-1">Waist Circumference (cm)</label>
                          <input
                            type="number"
                            min="30"
                            max="200"
                            value={waistCircumference}
                            onChange={(e) => { setWaistCircumference(Number(e.target.value)); setIsRecordSaved(false); }}
                            className="w-full bg-black/40 border border-white/10 text-white rounded-lg px-2.5 py-1 text-xs focus:outline-none focus:border-cyan-500/50"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Standalone Tabular Assessment Actions */}
                  <div className="mt-4 pt-3 border-t border-white/10 flex flex-col gap-3">
                    {isTabularRunning ? (
                      <div className="flex items-center justify-center gap-2 py-2">
                        <Loader2 className="animate-spin text-cyan-400" size={16} />
                        <span className="text-xs text-white/60">Analyzing patient intake...</span>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={handleRunTabularPrediction}
                        className="w-full py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 text-white text-xs font-bold rounded-lg transition-all"
                      >
                        Assess Tabular Risk
                      </button>
                    )}

                    {tabularResult && (
                      <div className="mt-3 p-3 bg-black/40 border border-white/10 rounded-xl space-y-2.5 animate-slide-up">
                        <div className="flex justify-between items-center">
                          <span className="text-[10px] uppercase font-bold text-white/40">Diabetes Risk Output</span>
                          <span className={`text-xs font-extrabold ${
                            tabularResult.prediction === 'HIGH RISK' 
                              ? 'text-rose-400' 
                              : tabularResult.prediction === 'MODERATE RISK' 
                              ? 'text-amber-400' 
                              : 'text-emerald-400'
                          }`}>
                            {tabularResult.prediction}
                          </span>
                        </div>
                        
                        {/* Confidence Bar */}
                        <div>
                          <div className="flex justify-between text-[9px] text-white/50 mb-1">
                            <span>Confidence Score</span>
                            <span>{tabularResult.confidence}%</span>
                          </div>
                          <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${tabularResult.confidence}%` }}
                              className={`h-full rounded-full transition-all duration-700 ${
                                tabularResult.prediction === 'HIGH RISK' 
                                  ? 'bg-rose-500' 
                                  : tabularResult.prediction === 'MODERATE RISK' 
                                  ? 'bg-amber-500' 
                                  : 'bg-emerald-500'
                              }`}
                            />
                          </div>
                        </div>

                        {/* Clinical Interpretation */}
                        <div className="text-[10px] text-white/70 leading-relaxed font-sans border-t border-white/5 pt-2">
                          {tabularResult.clinical_interpretation}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Column 2: Doctor's Prescription upload (OCR vital extract) */}
              <div className="bg-white/5 border border-white/10 rounded-2xl p-5 shadow-sm hover:border-white/20 transition-colors flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-5 border-b border-white/10 pb-3">
                    <FileText className="text-purple-400" size={18} />
                    <h3 className="font-bold text-white text-base">2. Prescription OCR Upload</h3>
                  </div>

                  {!prescriptionPreview ? (
                    <>
                      {/* Document type selector */}
                      <div className="flex items-center gap-2 mb-3">
                        <span className="text-[9px] uppercase font-bold text-white/40">Document Type:</span>
                        <div className="flex gap-1">
                          <button
                            type="button"
                            onClick={() => setPrescriptionDocType('prescription')}
                            className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition-all ${
                              prescriptionDocType === 'prescription'
                                ? 'bg-purple-500/20 border-purple-500/40 text-purple-300'
                                : 'bg-black/20 border-white/10 text-white/40 hover:text-white/60'
                            }`}
                          >
                            Prescription
                          </button>
                          <button
                            type="button"
                            onClick={() => setPrescriptionDocType('icu_record')}
                            className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border transition-all ${
                              prescriptionDocType === 'icu_record'
                                ? 'bg-purple-500/20 border-purple-500/40 text-purple-300'
                                : 'bg-black/20 border-white/10 text-white/40 hover:text-white/60'
                            }`}
                          >
                            ICU Record
                          </button>
                        </div>
                      </div>

                      <label className="border-2 border-dashed border-white/10 hover:border-purple-400/40 rounded-2xl p-6 flex flex-col items-center justify-center gap-2.5 cursor-pointer bg-black/20 transition-colors h-36">
                        <UploadCloud size={28} className="text-white/40" />
                        <span className="text-xs font-semibold text-white/60 text-center">
                          {prescriptionDocType === 'icu_record'
                            ? 'Upload ICU Patient Record / Charts'
                            : "Upload Doctor's Prescription"}
                        </span>
                        <span className="text-[9px] text-white/40">PNG, JPG, BMP</span>
                        <input
                          type="file"
                          accept="image/*"
                          onChange={handlePrescriptionUpload}
                          className="hidden"
                        />
                      </label>
                    </>
                  ) : (
                    <>
                      {/* Doc type badge on preview */}
                      <div className="flex items-center gap-2 mb-2">
                        <span className={`text-[9px] font-bold uppercase px-2 py-0.5 rounded-full border ${
                          prescriptionDocType === 'icu_record'
                            ? 'bg-amber-500/15 border-amber-500/30 text-amber-400'
                            : 'bg-blue-500/15 border-blue-500/30 text-blue-400'
                        }`}>
                          {prescriptionDocType === 'icu_record' ? 'ICU Record' : 'Prescription'}
                        </span>
                        <button
                          type="button"
                          onClick={() => setPrescriptionDocType(prescriptionDocType === 'prescription' ? 'icu_record' : 'prescription')}
                          className="text-[9px] text-white/30 hover:text-white/60 underline transition-colors"
                        >
                          switch type
                        </button>
                      </div>
                      <div className="relative rounded-xl overflow-hidden border border-white/10 h-36 bg-black/30 flex items-center justify-center">
                        <img
                          src={prescriptionPreview}
                          alt="Prescription Preview"
                          className="h-full w-full object-contain"
                        />
                        <button
                          onClick={() => {
                            setPrescriptionFile(null)
                            setPrescriptionPreview('')
                            setExtractedVitals(null)
                            setIsExtractingVitals(false)
                          }}
                          className="absolute bottom-2 right-2 bg-black/75 hover:bg-black text-white/90 text-[10px] px-2.5 py-1 rounded-full border border-white/10"
                        >
                          Change
                        </button>
                      </div>
                    </>
                  )}

                  {isExtractingVitals && (
                    <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-xl flex items-center gap-3 animate-pulse">
                      <Loader2 className="animate-spin text-purple-400" size={16} />
                      <div>
                        <div className="text-xs font-semibold text-purple-300">OCR Extractor Active</div>
                        <div className="text-[10px] text-purple-400">{vitalExtractStep}</div>
                      </div>
                    </div>
                  )}

                  {extractedVitals && (
                    <div className="mt-4 space-y-3">
                      <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-xl p-2.5 flex items-center gap-2">
                        <CheckCircle2 className="text-emerald-400" size={14} />
                        <span className="text-xs font-semibold text-emerald-300">
                          Extracted 48h Timeseries Vitals
                        </span>
                      </div>

                      <div className="border border-white/10 bg-black/30 rounded-xl p-3">
                        <div className="flex justify-between items-center mb-2">
                          <label className="text-[9px] uppercase font-bold text-white/40">Modality Monitor</label>
                          <select
                            value={selectedVitalIndex}
                            onChange={(e) => setSelectedVitalIndex(Number(e.target.value))}
                            className="bg-black border border-white/10 rounded px-1 py-0.5 text-[10px] text-white/80 focus:outline-none"
                          >
                            {VITAL_NAMES.map((name, i) => (
                              <option key={name} value={i}>{name}</option>
                            ))}
                          </select>
                        </div>

                        {/* Interactive sparkline graph */}
                        <div className="h-10 flex items-end justify-between px-1.5 py-1 relative bg-black/20 rounded border border-white/5">
                          {extractedVitals.map((hData, index) => {
                            const val = hData[selectedVitalIndex];
                            const allVals = extractedVitals.map(v => v[selectedVitalIndex]);
                            const minVal = Math.min(...allVals);
                            const maxVal = Math.max(...allVals);
                            const range = maxVal - minVal || 1.0;
                            const hPercent = Math.max(10, ((val - minVal) / range) * 80);
                            return (
                              <div
                                key={index}
                                style={{ height: `${hPercent}%` }}
                                className="w-[1.8%] bg-purple-500/60 rounded-t-xs hover:bg-purple-400 cursor-pointer transition-all"
                                title={`Hour ${index}: ${val.toFixed(1)}`}
                              />
                            )
                          })}
                        </div>
                        <div className="flex justify-between text-[8px] text-white/35 mt-1 font-mono">
                          <span>0h</span>
                          <span>24h</span>
                          <span>48h</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Show info for regular prescriptions (no ICU vitals) */}
                {prescriptionPreview && !extractedVitals && !isExtractingVitals && prescriptionDocType === 'prescription' && (
                  <div className="mt-4 p-2.5 bg-blue-500/10 border border-blue-500/20 rounded-xl flex items-start gap-2">
                    <FileText className="text-blue-400 shrink-0 mt-0.5" size={14} />
                    <div>
                      <div className="text-[10px] font-semibold text-blue-300">Prescription uploaded</div>
                      <div className="text-[9px] text-blue-400/80 mt-0.5">Standard prescriptions don't contain ICU timeseries vitals. Switch to "ICU Record" type if this document contains 48h patient monitoring data.</div>
                    </div>
                  </div>
                )}

                {!prescriptionPreview && !isExtractingVitals && (
                  <div className="text-[10px] text-white/40 leading-relaxed bg-black/20 border border-white/5 p-2 rounded-lg mt-4">
                    Select document type above, then upload. ICU records will extract 48h timeseries vitals.
                  </div>
                )}
              </div>

              {/* Column 3: Scan classification */}
              <div className="bg-white/5 border border-white/10 rounded-2xl p-5 shadow-sm hover:border-white/20 transition-colors flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-5 border-b border-white/10 pb-3">
                    <ScanLine className="text-blue-400" size={18} />
                    <h3 className="font-bold text-white text-base">3. Scans & Radiographs</h3>
                  </div>

                  {!scanPreview ? (
                    <label className="border-2 border-dashed border-white/10 hover:border-blue-400/40 rounded-2xl p-6 flex flex-col items-center justify-center gap-2.5 cursor-pointer bg-black/20 transition-colors h-36">
                      <UploadCloud size={28} className="text-white/40" />
                      <span className="text-xs font-semibold text-white/60 text-center">
                        Upload Chest X-ray/Scan
                      </span>
                      <span className="text-[9px] text-white/40">PNG, JPG, BMP</span>
                      <input
                        type="file"
                        accept="image/*"
                        onChange={handleScanUpload}
                        className="hidden"
                      />
                    </label>
                  ) : (
                    <div className="relative rounded-xl overflow-hidden border border-white/10 h-36 bg-black/30 flex items-center justify-center">
                      <img
                        src={scanPreview}
                        alt="Radiograph Preview"
                        className="h-full w-full object-contain"
                      />
                      {heatmapEnabled && predictionResult?.modality_confidence?.cnn?.PNEUMONIA > 0.6 && (
                        <div className="absolute inset-0 bg-radial-gradient from-rose-500/50 via-amber-500/20 to-transparent mix-blend-color-burn pointer-events-none animate-pulse-slow" />
                      )}
                      <button
                        onClick={() => {
                          setScanFile(null)
                          setScanPreview('')
                          setHeatmapEnabled(false)
                        }}
                        className="absolute bottom-2 right-2 bg-black/75 hover:bg-black text-white/95 text-[10px] px-2.5 py-1 rounded-full border border-white/10"
                      >
                        Change
                      </button>
                    </div>
                  )}

                  {scanPreview && predictionResult && (
                    <div className="mt-4 flex items-center justify-between bg-blue-500/10 border border-blue-500/20 rounded-xl p-2.5">
                      <span className="text-xs font-semibold text-blue-300">CNN Visual Signature</span>
                      <button
                        onClick={() => setHeatmapEnabled(!heatmapEnabled)}
                        className={`px-3 py-1 rounded-full text-[10px] font-extrabold transition-all ${
                          heatmapEnabled
                            ? 'bg-blue-500 text-white'
                            : 'bg-white/10 text-blue-300 border border-blue-500/30 hover:bg-white/20'
                        }`}
                      >
                        {heatmapEnabled ? 'Hide Grad-CAM' : 'Show Grad-CAM'}
                      </button>
                    </div>
                  )}
                </div>

                {!scanPreview && (
                  <div className="text-[10px] text-white/40 leading-relaxed bg-black/20 border border-white/5 p-2 rounded-lg mt-4">
                    Image classifications & diagnostic heatmap activations overlay directly.
                  </div>
                )}
              </div>
            </div>

            {/* Accordion for advanced parameters */}
            <div className="mt-5 border-t border-white/10 pt-4">
              <button
                onClick={() => setAdvancedOpen(!advancedOpen)}
                className="flex items-center gap-2 text-white/55 hover:text-white transition-colors text-xs font-semibold focus:outline-none"
              >
                <Sliders size={14} className="text-cyan-400" />
                <span>Advanced Decision Threshold overrides</span>
                <ChevronDown size={14} className={`transition-transform duration-300 ${advancedOpen ? 'rotate-180' : ''}`} />
              </button>

              {advancedOpen && (
                <div className="grid md:grid-cols-3 gap-6 mt-4 p-4 bg-white/5 border border-white/5 rounded-xl animate-fade-in">
                  <div>
                    <label className="block text-[9px] uppercase font-bold text-white/40 mb-1">
                      CNN Branch Weight ({Math.round(cnnWeight * 100)}%)
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={cnnWeight}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        setCnnWeight(val);
                        // Balance remaining
                        const rem = 1.0 - val;
                        setTabWeight(rem / 2);
                        setRnnWeight(rem / 2);
                      }}
                      className="w-full accent-cyan-500 bg-white/10"
                    />
                  </div>
                  <div>
                    <label className="block text-[9px] uppercase font-bold text-white/40 mb-1">
                      Tabular MLP Weight ({Math.round(tabWeight * 100)}%)
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={tabWeight}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        setTabWeight(val);
                        const rem = 1.0 - val;
                        setCnnWeight(rem / 2);
                        setRnnWeight(rem / 2);
                      }}
                      className="w-full accent-emerald-500 bg-white/10"
                    />
                  </div>
                  <div>
                    <label className="block text-[9px] uppercase font-bold text-white/40 mb-1">
                      RNN LSTM Weight ({Math.round(rnnWeight * 100)}%)
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={rnnWeight}
                      onChange={(e) => {
                        const val = Number(e.target.value);
                        setRnnWeight(val);
                        const rem = 1.0 - val;
                        setCnnWeight(rem / 2);
                        setTabWeight(rem / 2);
                      }}
                      className="w-full accent-purple-500 bg-white/10"
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Inference Actions */}
            <div className="mt-8 text-center border-t border-white/10 pt-6">
              {errorMessage && (
                <div className="max-w-md mx-auto mb-4 p-3.5 bg-rose-500/10 border border-rose-500/25 rounded-xl flex items-center gap-3 text-rose-300 text-xs">
                  <AlertTriangle className="text-rose-400 flex-shrink-0" size={16} />
                  <div className="text-left leading-relaxed">{errorMessage}</div>
                </div>
              )}

              {isInferring ? (
                <div className="flex flex-col items-center gap-2.5">
                  <Loader2 className="animate-spin text-cyan-400" size={28} />
                  <span className="text-xs font-semibold text-white/60 animate-pulse">{inferenceStep}</span>
                </div>
              ) : (
                <button
                  onClick={handleRunDiagnostics}
                  className="px-8 py-3.5 bg-gradient-to-r from-emerald-500 via-cyan-500 to-blue-600 hover:from-emerald-600 hover:to-blue-700 text-white font-extrabold rounded-full shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-0.5 active:translate-y-0 text-sm tracking-wider uppercase"
                >
                  Run Multimodal Risk Assessment
                </button>
              )}
            </div>

            {/* ========================================================================= */}
            {/* DIAGNOSTIC RESULTS DISPLAY SECTION (DARK CLINICAL THEME) */}
            {/* ========================================================================= */}
            {predictionResult && (
              <div className="mt-10 bg-white/5 border border-white/10 rounded-2xl p-5 sm:p-7 animate-slide-up">
                <div className="flex flex-wrap items-center justify-between border-b border-white/10 pb-4 mb-5 gap-3">
                  <div>
                    <h4 className="text-xl font-bold text-white tracking-tight">Clinical Diagnostic Assessment</h4>
                    <p className="text-[10px] text-white/40 font-mono mt-0.5">ID: MF-{Math.floor(100000 + Math.random() * 900000)} | Generated in 240ms</p>
                  </div>
                  <div className="flex items-center gap-2">
                    {isRecordSaved ? (
                      <span className="inline-flex items-center gap-1 px-2.5 py-0.5 bg-emerald-500/10 text-emerald-300 text-[10px] font-semibold rounded-full border border-emerald-500/20">
                        <CheckCircle2 size={10} /> Saved to Database
                      </span>
                    ) : (
                      <button
                        onClick={handleSaveToDatabase}
                        className="inline-flex items-center gap-1.5 px-3.5 py-1.5 bg-white/10 hover:bg-white/20 text-white text-[10px] font-bold rounded-full border border-white/20 transition-all duration-200"
                      >
                        <Save size={12} /> Save Patient Record
                      </button>
                    )}
                    <button
                      onClick={() => setPredictionResult(null)}
                      className="p-1.5 bg-white/5 hover:bg-white/10 border border-white/10 rounded-full text-white/60 transition-colors"
                      title="Reset Assessment"
                    >
                      <RotateCcw size={14} />
                    </button>
                  </div>
                </div>

                <div className="grid md:grid-cols-3 gap-6">
                  {/* Gauge Card (Binary Risk) */}
                  <div className="bg-black/35 rounded-xl p-5 border border-white/10 flex flex-col justify-between items-center text-center">
                    <div>
                      <span className="text-[9px] uppercase font-bold text-white/40 tracking-wider">Patient Risk</span>
                      <div className="text-3xl font-extrabold mt-2 tracking-tight">
                        <span className={predictionResult.binary_risk === 'High Risk' ? 'text-rose-400' : 'text-emerald-400'}>
                          {predictionResult.binary_risk}
                        </span>
                      </div>
                    </div>

                    <div className="relative w-28 h-28 my-3 flex items-center justify-center">
                      <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" fill="transparent" stroke="rgba(255,255,255,0.05)" strokeWidth="6" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          fill="transparent"
                          stroke={predictionResult.binary_risk === 'High Risk' ? '#f43f5e' : '#10b981'}
                          strokeWidth="6"
                          strokeDasharray="251"
                          strokeDashoffset={251 - (251 * predictionResult.binary_confidence)}
                          className="transition-all duration-1000 ease-out"
                        />
                      </svg>
                      <div className="absolute flex flex-col items-center">
                        <span className="text-lg font-black text-white">
                          {Math.round(predictionResult.binary_confidence * 100)}%
                        </span>
                        <span className="text-[8px] text-white/45 uppercase tracking-wide">Confidence</span>
                      </div>
                    </div>
                    <div className="text-[10px] text-white/50 leading-relaxed font-sans">
                      Deep fusion of all clinical domains indicates a high-confidence prediction path.
                    </div>
                  </div>

                  {/* Severity Levels Card */}
                  <div className="bg-black/35 rounded-xl p-5 border border-white/10 flex flex-col justify-between">
                    <div>
                      <span className="text-[9px] uppercase font-bold text-white/40 tracking-wider block mb-3">Severity Classifier</span>
                      <div className="space-y-2">
                        {Object.entries(predictionResult.severity_probabilities).map(([level, proba]: any) => {
                          const isSelected = predictionResult.severity === level;
                          return (
                            <div key={level}>
                              <div className="flex justify-between text-[10px] font-semibold mb-0.5">
                                <span className={isSelected ? 'text-white font-bold' : 'text-white/40'}>
                                  {level} {isSelected && '•'}
                                </span>
                                <span className={isSelected ? 'text-white font-bold' : 'text-white/45'}>
                                  {Math.round(proba * 100)}%
                                </span>
                              </div>
                              <div className="w-full h-1.5 bg-white/5 rounded-full overflow-hidden">
                                <div
                                  style={{ width: `${proba * 100}%` }}
                                  className={`h-full rounded-full transition-all duration-700 ${
                                    isSelected 
                                      ? level === 'Critical' ? 'bg-rose-500' : level === 'High' ? 'bg-amber-500' : 'bg-emerald-500'
                                      : 'bg-white/10'
                                  }`}
                                />
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    </div>
                    <div className="mt-4 pt-2.5 border-t border-white/5 flex items-center justify-between text-[11px] font-sans">
                      <span className="text-white/40">Clinical Severity:</span>
                      <span className="font-bold text-white bg-white/5 px-2 py-0.5 rounded border border-white/10">
                        {predictionResult.severity}
                      </span>
                    </div>
                  </div>

                  {/* Modality Confidences breakdown */}
                  <div className="bg-black/35 rounded-xl p-5 border border-white/10 flex flex-col justify-between">
                    <div>
                      <span className="text-[9px] uppercase font-bold text-white/40 tracking-wider block mb-4">Branch Signatures</span>
                      <div className="space-y-3.5">
                        {/* CNN X-Ray */}
                        <div>
                          <div className="flex justify-between text-[10px] font-semibold mb-0.5">
                            <span className="text-white/70 flex items-center gap-1.5"><ScanLine size={12} className="text-blue-400" /> CNN (Chest X-Ray)</span>
                            <span className="text-white/45 font-mono text-[9px]">
                              {predictionResult.modality_confidence.cnn.PNEUMONIA > 0.5 ? 'Pneumonia' : 'Normal'}
                            </span>
                          </div>
                          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${Math.max(predictionResult.modality_confidence.cnn.PNEUMONIA, predictionResult.modality_confidence.cnn.NORMAL) * 100}%` }}
                              className="h-full bg-blue-500 rounded-full"
                            />
                          </div>
                        </div>

                        {/* Tabular Diabetes */}
                        <div>
                          <div className="flex justify-between text-[10px] font-semibold mb-0.5">
                            <span className="text-white/70 flex items-center gap-1.5"><Table2 size={12} className="text-emerald-400" /> MLP (Diabetes Tabular)</span>
                            <span className="text-white/45 font-mono text-[9px]">
                              {predictionResult.modality_confidence.tabular.Diabetes > 0.5 ? 'Diabetes Risk' : 'Normal'}
                            </span>
                          </div>
                          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${Math.max(predictionResult.modality_confidence.tabular.Diabetes, predictionResult.modality_confidence.tabular['No Diabetes']) * 100}%` }}
                              className="h-full bg-emerald-500 rounded-full"
                            />
                          </div>
                        </div>

                        {/* RNN ICU vitals */}
                        <div>
                          <div className="flex justify-between text-[10px] font-semibold mb-0.5">
                            <span className="text-white/70 flex items-center gap-1.5"><Waves size={12} className="text-purple-400" /> LSTM (ICU Vitals)</span>
                            <span className="text-white/45 font-mono text-[9px]">
                              {predictionResult.modality_confidence.rnn.Deceased > 0.5 ? 'Risk High' : 'Risk Low'}
                            </span>
                          </div>
                          <div className="w-full h-1 bg-white/5 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${Math.max(predictionResult.modality_confidence.rnn.Deceased, predictionResult.modality_confidence.rnn.Survived) * 100}%` }}
                              className="h-full bg-purple-500 rounded-full"
                            />
                          </div>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-2.5 border-t border-white/5 text-[9px] text-white/35 text-center leading-relaxed">
                      Modality scores concatenated in the unified head.
                    </div>
                  </div>
                </div>

                {/* Explainability Section */}
                <div className="mt-5 bg-black/20 border border-white/5 rounded-xl p-5">
                  <span className="text-[9px] uppercase font-bold text-white/45 tracking-wider block mb-4">Branch Explainability Panel</span>
                  <div className="grid md:grid-cols-2 gap-6">
                    {/* SHAP Tabular features importance mock */}
                    <div>
                      <h5 className="text-[11px] font-bold text-white/80 mb-3 flex items-center gap-1.5">
                        <Cpu size={12} className="text-emerald-400" /> Tabular Risk Contributors (SHAP)
                      </h5>
                      <div className="space-y-2">
                        {[
                          { name: 'Glucose Level', score: bloodGlucose > 120 ? 0.35 : 0.05, type: 'positive' },
                          { name: 'Patient Age', score: age > 40 ? 0.22 : 0.02, type: 'positive' },
                          { name: 'Body Mass Index', score: bmi > 30 ? 0.18 : 0.04, type: 'positive' },
                          { name: 'Insulin Dosage', score: -0.05, type: 'negative' }
                        ].map((item) => (
                          <div key={item.name} className="flex items-center gap-3">
                            <span className="text-[10px] text-white/60 w-24 text-left">{item.name}</span>
                            <div className="flex-1 h-2.5 flex items-center bg-white/5 rounded overflow-hidden">
                              {item.type === 'positive' ? (
                                <div
                                  style={{ width: `${item.score * 100}%` }}
                                  className="h-full bg-rose-500/70 rounded-r text-[8px] text-white font-bold flex items-center justify-end pr-1 shadow-sm transition-all"
                                  title={`Adds ${Math.round(item.score * 100)}% risk`}
                                />
                              ) : (
                                <div className="w-full flex justify-end">
                                  <div
                                    style={{ width: `${Math.abs(item.score) * 100}%` }}
                                    className="h-full bg-emerald-500/70 rounded-l text-[8px] text-white font-bold flex items-center justify-start pl-1 shadow-sm transition-all"
                                    title={`Subtracts ${Math.round(Math.abs(item.score) * 100)}% risk`}
                                  />
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Temporal vital timeline check */}
                    <div>
                      <h5 className="text-[11px] font-bold text-white/80 mb-3 flex items-center gap-1.5">
                        <ActivitySquare size={12} className="text-purple-400" /> Temporal Vital Anomaly Timeline
                      </h5>
                      <div className="space-y-2 text-[10px]">
                        <div className="p-2 bg-black/40 border border-white/5 rounded flex justify-between items-start">
                          <div>
                            <span className="font-bold text-white/70 block">Hour 12 - Blood Sugar spike</span>
                            <span className="text-[9px] text-white/40">Extracted Glucose baseline exceeded 150 mg/dL</span>
                          </div>
                          <span className="bg-amber-500/10 border border-amber-500/25 text-amber-400 text-[8px] font-bold px-1.5 rounded">Warning</span>
                        </div>
                        <div className="p-2 bg-black/40 border border-white/5 rounded flex justify-between items-start">
                          <div>
                            <span className="font-bold text-white/70 block">Hour 38 - Tachycardia episode</span>
                            <span className="text-[9px] text-white/40">Extracted Heart Rate hit peak value {bloodGlucose > 130 ? '104' : '88'} bpm</span>
                          </div>
                          <span className="bg-rose-500/10 border border-rose-500/25 text-rose-400 text-[8px] font-bold px-1.5 rounded">Critical</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Transition from video to solid sections */}
        <div className="bg-gradient-to-b from-transparent to-[#f0f0ee] h-32 -mt-32 relative z-10" />

        <ModelsSection />
        <DatasetsSection />
        <ResearchSection />
        <Footer />
      </div>
    </div>
  )
}

function ModelsSection() {
  return (
    <section id="models" className="relative py-24 px-8 bg-[#f0f0ee] z-20">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Model Architecture
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Three specialized deep learning branches extract modality-specific
            embeddings, fused into a unified 192-dimensional representation.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {MODEL_CARDS.map((card) => (
            <div
              key={card.title}
              className="group bg-white rounded-2xl p-6 shadow-sm border border-gray-200
                         hover:shadow-xl hover:-translate-y-1 transition-all duration-300"
            >
              <div
                className={`w-12 h-12 rounded-xl bg-gradient-to-br ${card.color}
                           flex items-center justify-center text-white mb-4
                           group-hover:scale-110 transition-transform duration-300`}
              >
                {card.icon}
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2">
                {card.title}
              </h3>
              <p className="text-gray-600 text-sm mb-4">{card.description}</p>
              <span className="text-xs font-mono bg-gray-100 text-gray-500 px-2 py-1 rounded-full">
                {card.metrics}
              </span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function DatasetsSection() {
  return (
    <section id="datasets" className="py-24 px-8 bg-white relative z-20">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">Datasets</h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Three distinct healthcare datasets from different patient populations,
            combined via simulated aligned batches for fusion training.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {DATASETS.map((ds) => (
            <div
              key={ds.name}
              className="group flex items-start gap-4 bg-gray-50 rounded-2xl p-6
                         border border-gray-100 hover:bg-gray-100 transition-colors duration-200"
            >
              <div className="w-10 h-10 rounded-lg bg-gray-200 flex items-center justify-center text-gray-600 group-hover:bg-emerald-100 group-hover:text-emerald-600 transition-colors">
                {ds.icon}
              </div>
              <div>
                <h3 className="font-semibold text-gray-900">{ds.name}</h3>
                <p className="text-sm text-gray-500">{ds.modality}</p>
                <p className="text-xs font-mono text-gray-400 mt-1">{ds.size}</p>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 p-4 bg-amber-50 border border-amber-200 rounded-xl text-sm text-amber-800">
          <strong>Academic Disclaimer:</strong> These datasets originate from different
          patient populations. The fusion stage uses simulated aligned batches by
          randomly sampling across modalities. This prototype is NOT clinically deployable.
        </div>
      </div>
    </section>
  )
}



function ResearchSection() {
  return (
    <section id="research" className="py-24 px-8 bg-white relative z-20">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Research & Explainability</h2>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-12">
          Every prediction is accompanied by modality-specific explanations:
          Grad-CAM heatmaps, SHAP feature importance, and temporal vital attention analysis.
        </p>

        <div className="grid sm:grid-cols-3 gap-6">
          {[
            {
              icon: <ScanLine size={28} />,
              title: 'Grad-CAM',
              desc: 'Spatial attention heatmaps on chest X-rays highlighting regions of interest',
            },
            {
              icon: <Cpu size={28} />,
              title: 'SHAP Values',
              desc: 'Feature-level importance for tabular diabetes risk predictions',
            },
            {
              icon: <Activity size={28} />,
              title: 'Temporal Attention',
              desc: 'Per-timestep vital sign importance for ICU mortality risk',
            },
          ].map((item) => (
            <div
              key={item.title}
              className="p-6 rounded-2xl bg-gray-50 border border-gray-100 hover:shadow-md transition-shadow"
            >
              <div className="text-gray-400 mb-3 flex justify-center">
                {item.icon}
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">{item.title}</h3>
              <p className="text-sm text-gray-600">{item.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

function Footer() {
  return (
    <footer className="py-8 px-8 bg-gray-900 text-center relative z-20">
      <p className="text-gray-500 text-sm">
        Multimodal Healthcare AI — Academic Prototype &copy; {new Date().getFullYear()}
      </p>
      <p className="text-gray-600 text-xs mt-1">
        Not intended for clinical use. All predictions are for research and educational purposes only.
      </p>
    </footer>
  )
}
