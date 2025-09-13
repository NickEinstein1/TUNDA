# 🚀 Deploying Tunda to Render

## Prerequisites
1. GitHub account
2. Render account (free at render.com)
3. Your Tunda code pushed to GitHub

## Step-by-Step Deployment

### 1. Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Create Render Service
1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Select your Tunda repository

### 3. Configure Render Settings
- **Name**: `tunda-voice-companion`
- **Environment**: `Python 3`
- **Build Command**: `chmod +x build.sh && ./build.sh`
- **Start Command**: `python app.py`
- **Instance Type**: `Free`

### 4. Environment Variables
Add these in Render dashboard:
- `PYTHON_VERSION`: `3.9.18`
- `PORT`: `8000` (Render will override this)

### 5. Deploy
Click "Create Web Service" and wait for deployment!

## 🎯 Expected Behavior
- **Build time**: 5-10 minutes (first time)
- **URL**: `https://your-app-name.onrender.com`
- **Features**: Web interface only (no voice input due to browser limitations)

## 🔧 Troubleshooting
- If build fails: Check logs in Render dashboard
- If app crashes: Models might be too large for free tier
- If slow: Free tier has limited resources

## 📝 Notes
- Free tier sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds to wake up
- Voice input requires HTTPS (works on Render)
- Some ML models are simplified for deployment
