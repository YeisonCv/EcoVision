from datetime import datetime, timezone
from . import db


class Prediction(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(80), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc), index=True)

    def as_dict(self):
        created = self.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        return {"id": self.id, "filename": self.filename, "category": self.category,
                "confidence": self.confidence, "created_at": created.astimezone(timezone.utc).isoformat()}
