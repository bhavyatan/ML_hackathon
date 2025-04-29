import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from scipy import signal

st.set_page_config(
    page_title="Epileptic Seizure Detection",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the pre-trained seizure detection model.
    
    Returns:
        Loaded TensorFlow model or None if loading fails
    """
    try:
        model = tf.keras.models.load_model('CHB_MIT_sz_detec_demo.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_eeg(data: np.ndarray, sampling_rate: int = 256) -> np.ndarray:
    """Preprocess EEG data with bandpass filtering and normalization.
    
    Args:
        data: Raw EEG data array
        sampling_rate: Sampling frequency in Hz
        
    Returns:
        Preprocessed EEG data array
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    if not isinstance(sampling_rate, (int, float)):
        raise TypeError("Sampling rate must be a number")
    if sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")
    nyquist = sampling_rate / 2
    b, a = signal.butter(4, [0.5/nyquist, 40/nyquist], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    
    normalized_data = (filtered_data - np.mean(filtered_data)) / np.std(filtered_data)
    return normalized_data

st.title("🧠 Epileptic Seizure Detection")
st.write("""
This application uses deep learning to detect epileptic seizures from EEG data.
Please upload your EEG data file for analysis.
""")


st.sidebar.title("Settings")
sampling_rate = st.sidebar.number_input("Sampling Rate (Hz)", min_value=100, max_value=1000, value=256)


model = load_model()

uploaded_file = st.file_uploader("Upload EEG Data", type=['csv', 'txt', 'npy', 'edf'])

if uploaded_file is not None:
    try:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  
        if file_size > 100:  
            st.error("File size too large. Please upload a file smaller than 100MB.")
            st.stop()

        try:
            if uploaded_file.name.endswith('.npy'):
                eeg_data = np.load(uploaded_file)
                if not isinstance(eeg_data, np.ndarray):
                    st.error("Invalid NPY file format. Please upload a valid EEG data file.")
                    st.stop()
                if eeg_data.ndim != 1:
                    st.error("EEG data must be a 1-dimensional array.")
                    st.stop()
            elif uploaded_file.name.endswith('.edf'):
                import mne
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                try:
                    raw = mne.io.read_raw_edf(tmp_file_path, preload=True)
                    if len(raw.ch_names) == 0:
                        st.error("No channels found in the EDF file.")
                        st.stop()
                    eeg_data = raw.get_data()[0]  
                    if len(eeg_data) == 0:
                        st.error("Selected channel contains no data.")
                        st.stop()
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
            else:
                eeg_data = np.loadtxt(uploaded_file)
        except ValueError as ve:
            st.error(f"Error loading text file: {str(ve)}. Please ensure the file contains valid numeric data.")
            st.stop()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
        
        st.subheader("Raw EEG Data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=eeg_data, mode='lines', name='Raw EEG'))
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        processed_data = preprocess_eeg(eeg_data, sampling_rate)
        
        st.subheader("Processed EEG Data")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=processed_data, mode='lines', name='Processed EEG'))
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        if model is not None:
            
            segment_size = 2048
            overlap = 512
            segments = []
            
            if len(processed_data) < segment_size:
                st.error("EEG data is too short for analysis. Minimum required length is 2048 samples.")
                st.stop()
            
            for i in range(0, len(processed_data) - segment_size + 1, segment_size - overlap):
                segment = processed_data[i:i + segment_size]
                if np.std(segment) > 0:  
                    segments.append(segment)
            
            if not segments:
                st.error("No valid segments found in the EEG data.")
                st.stop()
                
            segments = np.array(segments)
            
            try:
                data_for_model = segments.reshape(-1, 32, 64, 1)
            except ValueError as ve:
                st.error(f"Error preparing data for model: {str(ve)}. Please ensure the data format is correct.")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error during data preparation: {str(e)}")
                st.stop()

            if data_for_model.shape[1:] != (32, 64, 1):
                st.error(f"Invalid data shape {data_for_model.shape}. Expected shape: (N, 32, 64, 1)")
                st.stop()
            
            try:
                with st.spinner('Analyzing EEG data...'):
                    predictions = model.predict(data_for_model, batch_size=32)
                    
                if not isinstance(predictions, np.ndarray):
                    st.error("Model prediction failed: Invalid output format")
                    st.stop()
                if len(predictions) == 0:
                    st.error("Model prediction failed: Empty output")
                    st.stop()
                if np.any(np.isnan(predictions)):
                    st.error("Model prediction failed: Contains invalid values")
                    st.stop()
                    
                window_size = 5
                smoothed_predictions = np.convolve(predictions.flatten(), 
                                                  np.ones(window_size)/window_size, 
                                                  mode='valid')
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.stop()
                
            
            st.subheader("Detection Results")
            avg_prob = np.mean(smoothed_predictions)
            max_prob = np.max(smoothed_predictions)
            
            high_prob_segments = np.sum(smoothed_predictions > 0.5) / len(smoothed_predictions)
            
            st.subheader("Seizure Probability Timeline")
            timeline_fig = go.Figure()
            timeline_fig.add_trace(go.Scatter(
                y=smoothed_predictions,
                mode='lines+markers',
                name='Seizure Probability',
                hovertemplate='Segment %{x}<br>Probability: %{y:.2%}'
            ))
            timeline_fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Threshold (0.5)")
            timeline_fig.update_layout(
                height=300,
                yaxis_title="Probability",
                xaxis_title="Segment Number",
                showlegend=True
            )
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maximum Probability", f"{max_prob:.2%}")
            with col2:
                st.metric("Average Probability", f"{avg_prob:.2%}")
            with col3:
                st.metric("High Probability Segments", f"{high_prob_segments:.2%}")
            
            seizure_threshold = 0.5
            min_high_prob_segments = 0.1  
            
            if max_prob > seizure_threshold and high_prob_segments > min_high_prob_segments:
                st.error("⚠️ Seizure Detected")
                st.warning(f"Found seizure patterns in {high_prob_segments:.1%} of the EEG segments")
            else:
                st.success("✅ No Seizure Detected")
            
            # st.info("""
            # 💡 Interpretation:
            # - Maximum Probability: Highest seizure probability across all segments
            # - Average Probability: Mean probability across all segments
            # - High Probability Segments: Percentage of segments showing seizure patterns
            # - Detection Criteria: 
            #   1. At least one segment must show high probability (>50%)
            #   2. At least 10% of segments must show seizure patterns
            # """)

                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

