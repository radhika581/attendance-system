# Automated Attendance System (Face Recognition + Flask + SQLite/MySQL)

A REST API that registers people by face, then marks attendance from a
photo using classical computer vision (Haar Cascade detection + LBPH
recognition) — no dlib/face\_recognition build headaches, just
`opencv-contrib-python`.

## Architecture

```
Client (curl / Postman / simple frontend)
        |
        v
   Flask REST API  (app.py)
        |
   +----+----------------------+
   |                           |
face\_utils.py              database.py
(detect + recognize)       (SQLite: users, attendance)
```

* **Detection**: Haar Cascade finds the face bounding box in an uploaded image.
* **Recognition**: LBPH (Local Binary Patterns Histograms) — trained on the
cropped, grayscale, histogram-equalized face — predicts a user ID + a
confidence score (lower = better match, since it's a distance metric).
* **Persistence**: SQLite by default (zero setup). Swapping to MySQL later
only touches `database.py` — same SQL, different connector.

## Why LBPH instead of a deep CNN embedding model?

CNN-based face embeddings (FaceNet, ArcFace) need either a pretrained
model or a lot of images per class. LBPH trains from just a handful of
photos per person, which fits a college/office registration flow where
you can't ask every student for 50 photos. It's also lightweight enough
to retrain instantly whenever someone new registers.

## Setup

```bash
python -m venv venv
venv\\Scripts\\activate        # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python app.py
```

Server runs at `http://localhost:5000`.

## API Reference

### `POST /api/register`

`multipart/form-data`: `roll\_no`, `name`, `images` (2–10 face photos, repeat the `images` field per file)

```bash
curl -X POST http://localhost:5000/api/register \\
  -F "roll\_no=CS101" -F "name=Radhika" \\
  -F "images=@face1.jpg" -F "images=@face2.jpg" -F "images=@face3.jpg"
```

Detects a face in each photo, saves the crops, and retrains the LBPH
model on all faces seen so far. If no face is found in any uploaded
photo, the user record is fully rolled back (no orphan rows).

### `POST /api/mark-attendance`

`multipart/form-data`: `image` (single photo, e.g. a webcam capture)

```bash
curl -X POST http://localhost:5000/api/mark-attendance -F "image=@capture.jpg"
```

Returns the matched user + confidence, or a 404 if no known face is
recognized. One attendance mark per person per day is enforced at the
DB layer (`UNIQUE(user\_id, date)`), so re-marking the same day returns
409 instead of a duplicate row.

### `GET /api/attendance?date=YYYY-MM-DD`

Attendance list for a specific date.

### `GET /api/attendance/summary`

Total days present per registered user.

### `GET /api/users`

All registered users.

## Database Schema

```sql
users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  roll\_no TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  created\_at TEXT NOT NULL
)

attendance (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user\_id INTEGER NOT NULL REFERENCES users(id),
  date TEXT NOT NULL,
  time TEXT NOT NULL,
  timestamp TEXT NOT NULL,
  UNIQUE(user\_id, date)
)
```

Indexed on `attendance(date)` for fast per-day report queries.

## Things worth extending (good "future work" talking points)

* Swap SQLite for MySQL (`mysql-connector-python`) for a multi-user deployment.
* Add a `/api/attendance/export` endpoint that streams a CSV.
* Add a lightweight HTML page with a webcam capture button instead of curl/Postman.
* Wrap Haar Cascade with a confidence-based re-check: if two people score
close confidences, ask for a second photo instead of guessing.
* Add JWT auth so only an admin can hit `/api/register`.

## What to say in interviews

* **OOP**: no formal classes here beyond what's needed, but you can point to
clear separation of concerns (`database.py` = persistence layer,
`face\_utils.py` = CV layer, `app.py` = controller/routing layer) as an
applied SRP (Single Responsibility Principle) example.
* **DBMS**: `UNIQUE(user\_id, date)` is enforcing a business rule
(one attendance per day) at the database level rather than in application
code — ask them why that's more robust (race conditions, multiple app
instances).
* **REST design**: idempotency angle — marking attendance twice in a day is
handled with a 409 Conflict, not a silent duplicate insert.

