# Update Google Drive URLs

After creating and uploading your pickle files:

1. **Get Google Drive File IDs:**
   - Right-click each uploaded .pkl file → Share → Copy link
   - Extract the file ID from URLs like: `https://drive.google.com/file/d/FILE_ID/view`

2. **Update model_downloader.py:**
   
Replace lines 15-16 with your actual file IDs:

```python
"distilbert_v2": "https://drive.google.com/uc?id=YOUR_DISTILBERT_FILE_ID&export=download",
"roberta": "https://drive.google.com/uc?id=YOUR_ROBERTA_FILE_ID&export=download"
```

3. **Commit and push:**
```bash
git add model_downloader.py
git commit -m "Update Google Drive URLs for pickle models"  
git push origin main
```

Then your deployment should show:
```
✅ Downloaded xgboost
✅ Downloaded svm
✅ Downloaded distilbert_v2  
✅ Downloaded roberta
✅ Ready with 4/4 working model(s): ['xgboost', 'svm', 'distilbert_v2', 'roberta']
```