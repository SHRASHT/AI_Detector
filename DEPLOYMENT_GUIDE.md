# 🚀 Deployment Guide: Streamlit Community Cloud

## Prerequisites
- GitHub account
- Streamlit Community Cloud account (free at share.streamlit.io)

## Step-by-Step Deployment

### 1. Create GitHub Repository
1. Go to [GitHub](https://github.com) and create a new repository
2. Name it: `ai-deepfake-detector` (or your preferred name)
3. Make it **Public** (required for Streamlit Community Cloud free tier)
4. Don't initialize with README (we have our own)

### 2. Upload Your Code to GitHub

#### Option A: Using GitHub Desktop (Easiest)
1. Download [GitHub Desktop](https://desktop.github.com/)
2. Clone your empty repository locally
3. Copy all files from `AI_Detector/` folder to the cloned repository
4. Commit and push to GitHub

#### Option B: Using Git Command Line
```bash
# Navigate to your AI_Detector folder
cd "d:\Shreshth WebApps\AI_Detector"

# Initialize git (if not done already)
git init

# Add remote repository (replace with your GitHub repo URL)
git remote add origin https://github.com/YourUsername/ai-deepfake-detector.git

# Add all files
git add .

# Commit
git commit -m "Initial commit: AI Deepfake Detector with feedback learning"

# Push to GitHub
git push -u origin main
```

#### Option C: Using GitHub Web Interface
1. Go to your empty GitHub repository
2. Click "uploading an existing file"
3. Drag and drop all files from your `AI_Detector` folder
4. Commit the files

### 3. Deploy to Streamlit Community Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)

2. **Sign In**: Use your GitHub account to sign in

3. **Create New App**:
   - Click "New app"
   - Repository: Select your `ai-deepfake-detector` repository
   - Branch: `main` (or `master`)
   - Main file path: `app.py`
   - App URL: Choose a custom URL (e.g., `your-username-deepfake-detector`)

4. **Deploy**: Click "Deploy!"

5. **Wait**: Deployment takes 2-5 minutes. Streamlit will:
   - Install all packages from `requirements.txt`
   - Set up the environment
   - Launch your app

### 4. Configuration (If Needed)

The app is pre-configured for Streamlit Cloud with:
- ✅ CPU-only PyTorch versions (cloud-compatible)
- ✅ Headless OpenCV (no display dependencies)
- ✅ Optimized memory usage
- ✅ Automatic dependency management

### 5. Monitor Deployment

In Streamlit Cloud dashboard, you can:
- View deployment logs
- Monitor app performance  
- Update settings
- Manage secrets (if needed)

### 6. Update Your App

To update your deployed app:
1. Make changes to your local code
2. Push changes to GitHub
3. Streamlit Cloud automatically redeploys

## 🎯 Streamlit Cloud Features

### ✅ What's Included (Free Tier):
- 1 GB RAM per app
- Automatic SSL (HTTPS)
- Custom domain support
- GitHub integration
- Automatic redeployments
- Community support

### ⚠️ Limitations:
- Public repositories only (free tier)
- Limited compute resources
- No persistent storage (files reset on reboot)
- Shared resources with other users

## 🔧 Troubleshooting Deployment

### Common Issues:

1. **Package Installation Fails**:
   - Check `requirements.txt` format
   - Ensure versions are compatible
   - Try removing version pinning for problematic packages

2. **Memory Errors**:
   - Streamlit Cloud has 1GB RAM limit
   - Model is optimized for CPU usage
   - Should work fine with current configuration

3. **File Not Found Errors**:
   - Check file paths are relative (not absolute)
   - Ensure all required files are in repository
   - Sample images should be in `data/` folder

4. **Import Errors**:
   - Verify all dependencies are in `requirements.txt`
   - Check Python version compatibility

### Logs and Debugging:
- View logs in Streamlit Cloud console
- Check deployment status
- Monitor resource usage

## 🚀 Advanced Configuration

### Custom Domain (Optional):
1. In Streamlit Cloud, go to app settings
2. Add custom domain
3. Configure DNS records

### Environment Variables:
- Use `.streamlit/secrets.toml` for sensitive data
- Add secrets in Streamlit Cloud dashboard
- Never commit secrets to GitHub

### Performance Optimization:
- Already optimized for cloud deployment
- Uses CPU-only PyTorch for efficiency
- Caches model loading with `@st.cache_resource`

## 🎉 Success!

Once deployed, your app will be available at:
`https://your-app-name.streamlit.app`

Share the link with others to let them try your AI Deepfake Detector!

## 📞 Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report issues in your repository

---

**Ready to deploy?** 🚀 Follow the steps above and your AI Detector will be live in minutes!
