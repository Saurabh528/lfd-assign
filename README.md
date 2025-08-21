# YOLOv8 OBB Lateral Flow Device Analyzer

A comprehensive Streamlit application for analyzing lateral flow devices (LFD) using YOLOv8 Oriented Bounding Box (OBB) detection and GPT-4o AI analysis.

## 🚀 Features

### **Core Detection Pipeline**
- **YOLOv8 OBB Detection**: Primary object detection using `best1.pt` model
- **Vertical Strip Detection**: Secondary detection using `vstrip.pt` model on cropped regions
- **Image Preprocessing**: Background normalization and enhancement
- **Heatmap Analysis**: Complete image processing pipeline with statistical analysis

### **AI-Powered Analysis**
- **GPT-4o Integration**: OpenAI's vision model for lateral flow analysis
- **Strict Classification**: STRONG/FAINT assessment for control and test lines
- **Multiple Test Lines**: Count and assess individual test line strengths
- **Consistent Templates**: Standardized image quality assessment

### **User Interface**
- **Interactive Streamlit App**: Modern web interface
- **Real-time Analysis**: Instant detection and analysis results
- **Download Capabilities**: Individual crops and ZIP bundles
- **Debug Information**: Raw JSON responses for troubleshooting

## 📋 Requirements

### **Python Dependencies**
```
streamlit>=1.48.1
ultralytics>=8.3.181
supervision>=0.26.1
opencv-python>=4.8.1.78
numpy>=1.26.4
pillow>=11.0.0
matplotlib>=3.10.5
scipy>=1.16.1
openai>=1.100.2
pandas>=2.3.1
```

### **Model Files**
- `runs/obb/train/weights/best1.pt` - Primary YOLOv8 OBB model
- `runs/obb/train/weights/vstrip.pt` - Vertical strip detection model

### **API Requirements**
- OpenAI API key for GPT-4o analysis

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd project_yolo
   ```

2. **Create conda environment**
   ```bash
   conda create -n yolo311 python=3.11
   conda activate yolo311
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## 🎯 Usage

### **Running the Application**
```bash
streamlit run streamlit_app.py
```

### **Application Workflow**

1. **Upload Image**: Select a lateral flow device image
2. **Configure Settings**: Adjust confidence thresholds and enable features
3. **Primary Detection**: YOLOv8 OBB detects objects using `best1.pt`
4. **Crop Analysis**: Each detection is cropped and analyzed individually
5. **AI Analysis**: Click "🤖 Analyze The Lines" for GPT-4o assessment
6. **Heatmap Analysis**: Click "🔍 Detect Heatmaps" for detailed processing
7. **Download Results**: Save individual crops or complete ZIP bundle

### **Analysis Features**

#### **GPT-4o Analysis**
- **Control Line Assessment**: Present/Absent with STRONG/FAINT classification
- **Test Line Assessment**: Count multiple lines with individual strength analysis
- **Result Classification**: POSITIVE/NEGATIVE/INVALID/AMBIGUOUS
- **Image Quality Notes**: Standardized quality assessment template

#### **Heatmap Analysis**
- **Image Processing Pipeline**: Grayscale, inversion, blur, thresholding
- **Statistical Analysis**: Intensity heatmaps, histograms, row-wise profiles
- **Component Analysis**: Connected components with area and aspect ratio filtering
- **Visualization**: Multiple plots showing processing steps

## 📊 Output Format

### **GPT-4o JSON Response**
```json
{
  "control_line": {
    "present": true,
    "strength": "STRONG"
  },
  "test_lines": {
    "present": false,
    "count": 0,
    "strengths": []
  },
  "result_classification": "NEGATIVE",
  "ambiguity_score": 0.2,
  "image_quality_notes": "Image Quality: GOOD | Lighting: GOOD | Blur: NONE | Notes: Clear image"
}
```

### **Display Metrics**
- **Control Line**: ✅ Present (STRONG) or ❌ Absent (N/A)
- **Test Lines**: ✅ 2 line(s) - STRONG, FAINT or ❌ Absent
- **Final Result**: 🟢 POSITIVE, 🔵 NEGATIVE, 🔴 INVALID, 🟡 AMBIGUOUS

## 🔧 Configuration

### **Sidebar Settings**
- **Model Path**: Customize YOLOv8 model location
- **Confidence Threshold**: Adjust detection sensitivity (0.60-1.0)
- **Device**: CPU or CUDA inference
- **Preprocessing**: Enable/disable image enhancement
- **OpenAI Settings**: API key, model selection, enable/disable

### **Analysis Parameters**
- **Crop Padding**: 10 pixels around detections
- **VStrip Confidence**: 0.25 threshold for secondary detection
- **Heatmap Processing**: 11x11 Gaussian blur, 92nd percentile threshold

## 📁 Project Structure

```
project_yolo/
├── streamlit_app.py          # Main Streamlit application
├── train_yolov8_obb.py       # YOLOv8 OBB training script
├── batch_process.py          # Batch processing script
├── streamlit_new_app.py      # Alternative two-model pipeline
├── vstrip_only_app.py        # VStrip-only analysis app
├── visualize.py              # Heatmap visualization logic
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── runs/
    └── obb/
        └── train/
            └── weights/
                ├── best1.pt  # Primary detection model
                └── vstrip.pt # Vertical strip model
```

## 🚀 Deployment

### **Local Development**
```bash
conda activate yolo311
streamlit run streamlit_app.py
```

### **Cloud Deployment**
- **Streamlit Cloud**: Direct deployment from GitHub
- **Heroku**: Container-based deployment
- **AWS/GCP**: Container orchestration

## 🔍 Troubleshooting

### **Common Issues**
1. **Model Loading Errors**: Ensure model files are in correct paths
2. **OpenAI API Errors**: Verify API key and quota
3. **Memory Issues**: Reduce image size or use CPU inference
4. **Detection Issues**: Adjust confidence thresholds

### **Debug Features**
- **Raw JSON Response**: Expand debug section to see GPT-4o output
- **Console Logs**: Check terminal for detailed error messages
- **Model Debug**: VStrip detection debug information

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: YOLOv8 framework
- **OpenAI**: GPT-4o vision model
- **Streamlit**: Web application framework
- **Supervision**: Computer vision utilities

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the debug information in the app
- Review the troubleshooting section

---

**Built with ❤️ for accurate lateral flow device analysis**
