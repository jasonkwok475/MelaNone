"""Patient request/response schemas."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class PatientBase(BaseModel):
    # De-identified label is required; a real name is optional (privacy-lean default).
    display_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=256)
    date_of_birth: date | None = None
    sex: str | None = Field(default=None, max_length=32)
    notes: str = ""
    consent_ack: bool = False


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    """All fields optional — only provided fields are changed."""

    display_id: str | None = Field(default=None, min_length=1, max_length=128)
    name: str | None = Field(default=None, max_length=256)
    date_of_birth: date | None = None
    sex: str | None = Field(default=None, max_length=32)
    notes: str | None = None
    consent_ack: bool | None = None


class PatientRead(PatientBase):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
    updated_at: datetime
