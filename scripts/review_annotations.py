"""Local bbox/class editor. Only explicit reviewed acceptance exports training labels."""
import argparse, json, secrets, threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
try:
    from .common import ROOT, timestamp, write_json
    from .custom_dataset import validate_boxes
except ImportError:
    from common import ROOT, timestamp, write_json
    from custom_dataset import validate_boxes

def create_app(folder):
    folder=Path(folder).resolve()
    manifest_path=folder/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    rows={r["id"]:r for r in manifest["images"]}
    app=FastAPI()
    token_path=folder/".review_token"
    token=token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else secrets.token_hex(24)
    token_path.write_text(token,encoding="utf-8")
    lock=threading.Lock()

    @app.get("/",response_class=HTMLResponse)
    def home():
        return (ROOT/"web/annotation_review.html").read_text(encoding="utf-8").replace("__REVIEW_TOKEN__",token)
    @app.get("/api/manifest")
    def get_manifest():
        return manifest
    @app.post("/api/import-new")
    def import_captures(request:Request):
        if request.headers.get("X-Review-Token")!=token: raise HTTPException(403,"Open the local review page")
        if "incremental_import" not in manifest: raise HTTPException(404,"Import not enabled for this frozen review")
        try: from .prepare_new_review import import_new
        except ImportError: from prepare_new_review import import_new
        with lock:
            try: result=import_new(folder,manifest)
            except ValueError as exc: raise HTTPException(409,str(exc)) from exc
            rows.update({r["id"]:r for r in manifest["images"]})
        return result

    @app.get("/image/{key}")
    def get_image(key):
        row=rows.get(key)
        if not row or not row.get("processed_file"): raise HTTPException(404,"Image unavailable")
        target=(folder/row["processed_file"]).resolve()
        if not target.is_relative_to(folder): raise HTTPException(400,"Invalid image path")
        return FileResponse(target,headers={"Cache-Control":"no-store"})
    @app.post("/api/review/{key}")
    async def review(key,request:Request):
        # Require an unguessable page token; no cross-origin write, no credentials/remote image upload.
        if request.headers.get("X-Review-Token")!=token: raise HTTPException(403,"Open the local review page")
        data=await request.json()
        row=rows.get(key)
        if not row or not row.get("processed_file"): raise HTTPException(404,"Image unavailable")
        action=data.get("action")
        if action not in {"accept","draft","exclude"}: raise HTTPException(400,"Invalid action")
        try:
            boxes=validate_boxes(data.get("boxes",[]),row["width"],row["height"],False) if action=="accept" else data.get("boxes",[])
        except (ValueError,TypeError) as exc: raise HTTPException(400,str(exc)) from exc
        if action=="accept" and row.get("face_count",0)>len(boxes):
            raise HTTPException(400,"Detector proposed more faces than reviewed. Keep/review all real faces; mark ambiguous images excluded.")
        reviewer=str(data.get("reviewer","")).strip()
        if action=="accept" and not reviewer: raise HTTPException(400,"Reviewer name required")
        with lock:
            row.update(boxes=boxes,label_status={"accept":"verified","draft":"draft","exclude":"excluded"}[action],
                       reviewer=reviewer,human_reviewed=action=="accept",reviewed_at=timestamp(),review_note=str(data.get("note",""))[:1000],included_in_training=False)
            manifest["statistics"]["verified_images"]=sum(r["label_status"]=="verified" for r in manifest["images"])
            pending=folder/".review_pending.json"
            write_json(pending,manifest)
            pending.replace(manifest_path)
        return {"ok":True,"label_status":row["label_status"]}
    return app

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--folder",type=Path,default=ROOT/"data/processed/single_person_review_v1")
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(args.folder),host="127.0.0.1",port=args.port)

if __name__=="__main__": main()
