import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from .session import Base

def utc_now():
    return datetime.now(timezone.utc)

class Client(Base):
    """Organizations / Ministries paying for and employing users."""
    __tablename__ = "clients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    billing_email = Column(String(255), nullable=True)
    subscription_tier = Column(String(100), default="Standard")
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    users = relationship("User", back_populates="client")
    jobs = relationship("Job", back_populates="client")


class User(Base):
    """User accounts (scientists, admins) running pipeline jobs."""
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    role = Column(String(100), default="scientist")
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    client = relationship("Client", back_populates="users")
    jobs = relationship("Job", back_populates="user")


class Region(Base):
    """Geographies & Administrative AOIs with PostGIS geometry."""
    __tablename__ = "regions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    geometry = Column(Geometry(geometry_type="MULTIPOLYGON", srid=4326), nullable=True)
    metadata_json = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    jobs = relationship("Job", back_populates="region")


class Job(Base):
    """Pipeline job execution tracking (operational & historical)."""
    __tablename__ = "jobs"

    id = Column(String(100), primary_key=True)  # supports UUID string or job_id
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    client_id = Column(UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL"), nullable=True)
    region_id = Column(UUID(as_uuid=True), ForeignKey("regions.id", ondelete="SET NULL"), nullable=True)
    
    module_name = Column(String(100), nullable=True, index=True)  # e.g., 'national landcover', 'sdg_11_3_1'
    status = Column(String(50), nullable=False, default="queued", index=True)
    credits_used = Column(Float, default=0.0)
    
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    # Full preserved scientific results & run parameters
    parameters = Column(JSONB, default=dict)
    result = Column(JSONB, default=dict)
    error = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="jobs")
    client = relationship("Client", back_populates="jobs")
    region = relationship("Region", back_populates="jobs")
    job_results = relationship("JobResult", back_populates="job", cascade="all, delete-orphan")

    def to_dict(self) -> dict:
        """Serialize to dictionary compatible with the frontend expectations."""
        # Determine clean country/AOI name from linked Region or parameters/results
        region_name = self.region.name if self.region else None
        if not region_name and self.parameters:
            region_name = (
                self.parameters.get("aoi_name")
                or self.parameters.get("country")
                or self.parameters.get("aoi_id")
            )
        if not region_name and self.result:
            region_name = (
                self.result.get("aoi_name")
                or self.result.get("country")
                or self.result.get("results", {}).get("inference", {}).get("aoi_name")
            )

        # Determine Indicator ID (e.g. TASKING for Land Cover, 11.3.1 for SDG 11.3.1)
        indicator = "11.3.1"
        if self.module_name in {"TASKING", "national landcover", "space_for_time_tasking"}:
            indicator = "TASKING"
        elif self.parameters and self.parameters.get("indicator_id"):
            indicator = self.parameters["indicator_id"]
        elif self.result and ("earth_engine" in str(self.result) or "inference" in str(self.result)):
            indicator = "TASKING"
        elif self.module_name:
            indicator = self.module_name

        params = self.parameters or {}

        return {
            "id": self.id,
            "job_id": self.id,
            "taskId": self.id,
            "module_name": self.module_name,
            "indicator_id": indicator,
            "aoi_name": region_name or "Global",
            "aoi_id": region_name or "Global",
            "country": region_name or "Global",
            "status": self.status,
            "state": (self.status or "").upper(),
            "credits_used": self.credits_used,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "parameters": params,
            "request": params,
            "result": self.result or {},
            "error": self.error,
        }



class JobResult(Base):
    """Granular scientific results produced by pipeline jobs."""
    __tablename__ = "job_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(100), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    result_type = Column(String(100), nullable=False, index=True)  # e.g., 'model_metrics', 'annual_sdg', 'inference'
    data = Column(JSONB, nullable=False, default=dict)  # Flexible science metrics
    created_at = Column(DateTime(timezone=True), default=utc_now)

    # Relationships
    job = relationship("Job", back_populates="job_results")
