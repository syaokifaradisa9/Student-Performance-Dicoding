import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Performa Mahasiswa", layout="wide")


@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(
            f"Error: File model tidak ditemukan di path '{path}'. Pastikan file 'xgb_model.model' disimpan menggunakan joblib.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat model: {e}")
        return None

model = load_model('xgb_model.model')

MODEL_FEATURE_ORDER = [
    'Marital_status', 'Application_mode', 'Application_order', 'Course',
    'Daytime_evening_attendance', 'Previous_qualification',
    'Previous_qualification_grade', 'Mothers_qualification',
    'Fathers_qualification', 'Mothers_occupation', 'Fathers_occupation',
    'Admission_grade', 'Displaced', 'Debtor', 'Tuition_fees_up_to_date',
    'Gender', 'Scholarship_holder', 'Age_at_enrollment',
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations', 'Unemployment_rate',
    'Inflation_rate', 'GDP'
]

marital_status_map = {'Single': 1, 'Married': 2, 'Widower': 3, 'Divorced': 4, 'Facto union': 5, 'Legally separated': 6}
application_mode_map = {'1st phase - general contingent': 1, 'Ordinance No. 612/93': 2,
                        '1st phase - special contingent (Azores Island)': 5, 'Holders of other higher courses': 7,
                        'Ordinance No. 854-B/99': 10, 'International student (bachelor)': 15,
                        '1st phase - special contingent (Madeira Island)': 16, '2nd phase - general contingent': 17,
                        '3rd phase - general contingent': 18, "Ordinance No. 533-A/99, item b2) (Different Plan)": 26,
                        "Ordinance No. 533-A/99, item b3 (Other Institution)": 27, 'Over 23 years old': 39,
                        'Transfer': 42, 'Change of course': 43, 'Technological specialization diploma holders': 44,
                        'Change of institution/course': 51, 'Short cycle diploma holders': 53,
                        'Change of institution/course (International)': 57}
course_map = {'Biofuel Production Technologies': 33, 'Animation and Multimedia Design': 171,
              'Social Service (evening attendance)': 8014, 'Agronomy': 9003, 'Communication Design': 9070,
              'Veterinary Nursing': 9085, 'Informatics Engineering': 9119, 'Equinculture': 9130, 'Management': 9147,
              'Social Service': 9238, 'Tourism': 9254, 'Nursing': 9500, 'Oral Hygiene': 9556,
              'Advertising and Marketing Management': 9670, 'Journalism and Communication': 9773,
              'Basic Education': 9853, 'Management (evening attendance)': 9991}
attendance_map = {'Daytime': 1, 'Evening': 0}
prev_qualification_map = {'Secondary education': 1, "Higher education - bachelor's degree": 2,
                          'Higher education - degree': 3, "Higher education - master's": 4,
                          'Higher education - doctorate': 5, 'Frequency of higher education': 6,
                          '12th year of schooling - not completed': 9, '11th year of schooling - not completed': 10,
                          'Other - 11th year of schooling': 12, '10th year of schooling': 14,
                          '10th year of schooling - not completed': 15,
                          'Basic education 3rd cycle (9th/10th/11th year) or equiv.': 19,
                          'Basic education 2nd cycle (6th/7th/8th year) or equiv.': 38,
                          'Technological specialization course': 39, 'Higher education - degree (1st cycle)': 40,
                          'Professional higher technical course': 42, 'Higher education - master (2nd cycle)': 43}

mother_qualification_map = {"Secondary Education - 12th Year of Schooling or Eq.": 1,
                            "Higher Education - Bachelor's Degree": 2, "Higher Education - Degree": 3,
                            "Higher Education - Master's": 4, "Higher Education - Doctorate": 5,
                            "Frequency of Higher Education": 6, "12th Year of Schooling - Not Completed": 9,
                            "11th Year of Schooling - Not Completed": 10, "7th Year (Old)": 11,
                            "Other - 11th Year of Schooling": 12, "10th Year of Schooling": 14,
                            "General commerce course": 18,
                            "Basic Education 3rd Cycle (9th/10th/11th Year) or Equiv.": 19,
                            "Technical-professional course": 22, "7th year of schooling": 26,
                            "2nd cycle of the general high school course": 27,
                            "9th Year of Schooling - Not Completed": 29, "8th year of schooling": 30, "Unknown": 34,
                            "Can't read or write": 35, "Can read without having a 4th year of schooling": 36,
                            "Basic education 1st cycle (4th/5th year) or equiv.": 37,
                            "Basic Education 2nd Cycle (6th/7th/8th Year) or Equiv.": 38,
                            "Technological specialization course": 39, "Higher education - degree (1st cycle)": 40,
                            "Specialized higher studies course": 41, "Professional higher technical course": 42,
                            "Higher Education - Master (2nd cycle)": 43, "Higher Education - Doctorate (3rd cycle)": 44}
father_qualification_map = mother_qualification_map.copy()
father_qualification_map.update(
    {"2nd year complementary high school course": 13, "Complementary High School Course": 20,
     "Complementary High School Course - not concluded": 25, "General Course of Administration and Commerce": 31,
     "Supplementary Accounting and Administration": 33})

mother_occupation_map = {"Student": 0,
                         "Representatives of the Legislative Power and Executive Bodies, Directors, Directors and Executive Managers": 1,
                         "Specialists in Intellectual and Scientific Activities": 2,
                         "Intermediate Level Technicians and Professions": 3, "Administrative staff": 4,
                         "Personal Services, Security and Safety Workers and Sellers": 5,
                         "Farmers and Skilled Workers in Agriculture, Fisheries and Forestry": 6,
                         "Skilled Workers in Industry, Construction and Craftsmen": 7,
                         "Installation and Machine Operators and Assembly Workers": 8, "Unskilled Workers": 9,
                         "Armed Forces Professions": 10, "Other Situation": 90, "(blank)": 99,
                         "Health professionals": 122, "Teachers": 123,
                         "Specialists in information and communication technologies (ICT)": 125,
                         "Intermediate level science and engineering technicians and professions": 131,
                         "Technicians and professionals, of intermediate level of health": 132,
                         "Intermediate level technicians from legal, social, sports, cultural and similar services": 134,
                         "Office workers, secretaries in general and data processing operators": 141,
                         "Data, accounting, statistical, financial services and registry-related operators": 143,
                         "Other administrative support staff": 144, "Personal service workers": 151, "Sellers": 152,
                         "Personal care workers and the like": 153,
                         "Skilled construction workers and the like, except electricians": 171,
                         "Skilled workers in printing, precision instrument manufacturing, jewelers, artisans and the like": 173,
                         "Workers in food processing, woodworking, clothing and other industries and crafts": 175,
                         "Cleaning workers": 191,
                         "Unskilled workers in agriculture, animal production, fisheries and forestry": 192,
                         "Unskilled workers in extractive industry, construction, manufacturing and transport": 193,
                         "Meal preparation assistants": 194}
father_occupation_map = mother_occupation_map.copy()
father_occupation_map.update(
    {"Armed Forces Officers": 101, "Armed Forces Sergeants": 102, "Other Armed Forces personnel": 103,
     "Directors of administrative and commercial services": 112,
     "Hotel, catering, trade and other services directors": 114,
     "Specialists in the physical sciences, mathematics, engineering and related techniques": 121,
     "Specialists in finance, accounting, administrative organization, public and commercial relations": 124,
     "Information and communication technology technicians": 135, "Protection and security services personnel": 154,
     "Market-oriented farmers and skilled agricultural and animal production workers": 161,
     "Farmers, livestock keepers, fishermen, hunters and gatherers, subsistence": 163,
     "Skilled workers in metallurgy, metalworking and similar": 172,
     "Skilled workers in electricity and electronics": 174, "Fixed plant and machine operators": 181,
     "Assembly workers": 182, "Vehicle drivers and mobile equipment operators": 183,
     "Street vendors (except food) and street service providers": 195})

yes_no_map = {'Yes': 1, 'No': 0}
gender_map = {'Male': 1, 'Female': 0}

st.title("Prediksi Performa Akademik Mahasiswa")
st.write("Aplikasi ini memprediksi status akademik mahasiswa (Lulus/Dropout) berdasarkan data lengkap.")

with st.form("prediction_form"):
    st.header("I. Informasi Pribadi & Pendaftaran")
    col1, col2 = st.columns(2)
    with col1:
        marital_status_in = st.selectbox("Status Pernikahan", options=list(marital_status_map.keys()))
        application_mode_in = st.selectbox("Jalur Pendaftaran", options=list(application_mode_map.keys()))
        course_in = st.selectbox("Program Studi", options=list(course_map.keys()))
        attendance_in = st.selectbox("Waktu Kuliah", options=list(attendance_map.keys()))
        gender_in = st.selectbox("Jenis Kelamin", options=list(gender_map.keys()))
        displaced_in = st.selectbox("Mahasiswa Pindahan (Displaced)", options=list(yes_no_map.keys()))
        debtor_in = st.selectbox("Memiliki Tunggakan", options=list(yes_no_map.keys()))

    with col2:
        application_order_in = st.number_input("Urutan Pilihan Aplikasi", min_value=0, max_value=9, value=1)
        age_in = st.number_input("Usia Saat Mendaftar", min_value=17, max_value=70, value=20)
        admission_grade_in = st.number_input("Nilai Pendaftaran", min_value=0.0, max_value=200.0, value=120.0)
        prev_qualification_in = st.selectbox("Kualifikasi Sebelumnya", options=list(prev_qualification_map.keys()))
        prev_qualification_grade_in = st.number_input("Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0,
                                                      value=120.0)
        tuition_fees_up_to_date_in = st.selectbox("Uang Kuliah Lunas", options=list(yes_no_map.keys()))
        scholarship_holder_in = st.selectbox("Penerima Beasiswa", options=list(yes_no_map.keys()))

    st.header("II. Informasi Orang Tua")
    col3, col4 = st.columns(2)
    with col3:
        mother_qualification_in = st.selectbox("Kualifikasi Ibu", options=list(mother_qualification_map.keys()))
        mother_occupation_in = st.selectbox("Pekerjaan Ibu", options=list(mother_occupation_map.keys()))
    with col4:
        father_qualification_in = st.selectbox("Kualifikasi Ayah", options=list(father_qualification_map.keys()))
        father_occupation_in = st.selectbox("Pekerjaan Ayah", options=list(father_occupation_map.keys()))

    st.header("III. Informasi Akademik Semester 1")
    col5, col6 = st.columns(2)
    with col5:
        curricular_units_1st_credited_in = st.number_input("SKS Diakui (Sem 1)", min_value=0, value=0)
        curricular_units_1st_enrolled_in = st.number_input("SKS Diambil (Sem 1)", min_value=0, value=6)
        curricular_units_1st_evaluations_in = st.number_input("Jumlah Evaluasi (Sem 1)", min_value=0, value=8)
    with col6:
        curricular_units_1st_approved_in = st.number_input("SKS Lulus (Sem 1)", min_value=0, value=5)
        curricular_units_1st_grade_in = st.number_input("Nilai Rata-rata (Sem 1)", min_value=0.0, max_value=20.0,
                                                        value=12.0)
        curricular_units_1st_without_evaluations_in = st.number_input("SKS Tanpa Evaluasi (Sem 1)", min_value=0,
                                                                      value=0)

    st.header("IV. Informasi Akademik Semester 2")
    col7, col8 = st.columns(2)
    with col7:
        curricular_units_2nd_credited_in = st.number_input("SKS Diakui (Sem 2)", min_value=0, value=0)
        curricular_units_2nd_enrolled_in = st.number_input("SKS Diambil (Sem 2)", min_value=0, value=6)
        curricular_units_2nd_evaluations_in = st.number_input("Jumlah Evaluasi (Sem 2)", min_value=0, value=8)
    with col8:
        curricular_units_2nd_approved_in = st.number_input("SKS Lulus (Sem 2)", min_value=0, value=5)
        curricular_units_2nd_grade_in = st.number_input("Nilai Rata-rata (Sem 2)", min_value=0.0, max_value=20.0,
                                                        value=12.0)
        curricular_units_2nd_without_evaluations_in = st.number_input("SKS Tanpa Evaluasi (Sem 2)", min_value=0,
                                                                      value=0)

    st.header("V. Data Ekonomi Makro")
    col9, col10, col11 = st.columns(3)
    with col9:
        unemployment_rate_in = st.number_input("Tingkat Pengangguran (%)", value=12.7, format="%.1f")
    with col10:
        inflation_rate_in = st.number_input("Tingkat Inflasi (%)", value=0.5, format="%.1f")
    with col11:
        gdp_in = st.number_input("GDP", value=1.79, format="%.2f")

    submitted = st.form_submit_button("SUBMIT UNTUK PREDIKSI")

if submitted:
    if model is not None:
        input_data = {
            'Marital_status': marital_status_map[marital_status_in],
            'Application_mode': application_mode_map[application_mode_in],
            'Application_order': application_order_in,
            'Course': course_map[course_in],
            'Daytime_evening_attendance': attendance_map[attendance_in],
            'Previous_qualification': prev_qualification_map[prev_qualification_in],
            'Previous_qualification_grade': prev_qualification_grade_in,
            'Mothers_qualification': mother_qualification_map[mother_qualification_in],
            'Fathers_qualification': father_qualification_map[father_qualification_in],
            'Mothers_occupation': mother_occupation_map[mother_occupation_in],
            'Fathers_occupation': father_occupation_map[father_occupation_in],
            'Admission_grade': admission_grade_in,
            'Displaced': yes_no_map[displaced_in],
            'Debtor': yes_no_map[debtor_in],
            'Tuition_fees_up_to_date': yes_no_map[tuition_fees_up_to_date_in],
            'Gender': gender_map[gender_in],
            'Scholarship_holder': yes_no_map[scholarship_holder_in],
            'Age_at_enrollment': age_in,
            'Curricular_units_1st_sem_credited': curricular_units_1st_credited_in,
            'Curricular_units_1st_sem_enrolled': curricular_units_1st_enrolled_in,
            'Curricular_units_1st_sem_evaluations': curricular_units_1st_evaluations_in,
            'Curricular_units_1st_sem_approved': curricular_units_1st_approved_in,
            'Curricular_units_1st_sem_grade': curricular_units_1st_grade_in,
            'Curricular_units_1st_sem_without_evaluations': curricular_units_1st_without_evaluations_in,
            'Curricular_units_2nd_sem_credited': curricular_units_2nd_credited_in,
            'Curricular_units_2nd_sem_enrolled': curricular_units_2nd_enrolled_in,
            'Curricular_units_2nd_sem_evaluations': curricular_units_2nd_evaluations_in,
            'Curricular_units_2nd_sem_approved': curricular_units_2nd_approved_in,
            'Curricular_units_2nd_sem_grade': curricular_units_2nd_grade_in,
            'Curricular_units_2nd_sem_without_evaluations': curricular_units_2nd_without_evaluations_in,
            'Unemployment_rate': unemployment_rate_in,
            'Inflation_rate': inflation_rate_in,
            'GDP': gdp_in,
        }

        input_df_raw = pd.DataFrame([input_data])

        try:
            input_df = input_df_raw[MODEL_FEATURE_ORDER]

            prediction = model.predict(input_df)
            prediction_proba = model.predict_proba(input_df)

            target_map = {0: 'Lulus', 1: 'Dropout', 2: 'Enrolled'}
            result = target_map.get(prediction[0], "Tidak Diketahui")

            st.subheader("Hasil Prediksi:")

            if result == 'Lulus':
                st.success(f"✅ Mahasiswa diprediksi akan **LULUS**.")
            else:
                st.warning(f"⚠️ Mahasiswa diprediksi akan **DROPOUT**.")

            st.write(f"Probabilitas (Kepercayaan Model):")
            st.write(f"- **Lulus**: {prediction_proba[0][0]:.2%}")
            st.write(f"- **Dropout**: {prediction_proba[0][1]:.2%}")
            st.write(f"- **Enrolled**: {prediction_proba[0][2]:.2%}")

        except KeyError as e:
            st.error(f"Error: Kolom yang dibutuhkan oleh model tidak ditemukan: {e}")
            st.error(
                "Pastikan daftar `MODEL_FEATURE_ORDER` di dalam kode sudah benar dan sesuai dengan nama kolom saat training.")
        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.error("Pastikan file model `xgb_model.model` kompatibel dan dilatih dengan fitur yang sama.")