# Database Integration Plan for FieldOS Engine

This document outlines the plan to integrate a database into FieldOS Engine, enabling a customizable frontend where users can create, edit, and manage all aspects of the system.

## Overview

Currently, FieldOS Engine uses JSON files for data storage and an in-memory registry. This plan migrates to a proper database while maintaining backward compatibility with existing JSON imports.

---

## Phase 1: Database Setup & Core Models

### 1.1 Choose Database

**Recommended: PostgreSQL with SQLAlchemy ORM**

Reasons:
- Robust JSON/JSONB support for flexible fields (routes, positions)
- Strong typing with good Python integration
- Scales well for multi-tenant (multiple teams/users)
- SQLAlchemy provides async support via `asyncpg`

Alternative for simpler deployments: SQLite (same SQLAlchemy code works)

### 1.2 Database Schema

```sql
-- Users and Teams
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE teams (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    owner_id UUID REFERENCES users(id),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE team_members (
    team_id UUID REFERENCES teams(id),
    user_id UUID REFERENCES users(id),
    role VARCHAR(50) DEFAULT 'member',  -- 'owner', 'coach', 'member'
    PRIMARY KEY (team_id, user_id)
);

-- Core Game Entities
CREATE TABLE players (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    external_id VARCHAR(100),  -- e.g., "player_wr1_1"
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,  -- QB, CENTER, WR1, WR2, WR3, D1-D5
    speed FLOAT DEFAULT 4.5,
    acceleration FLOAT DEFAULT 2.0,
    route_sharpness FLOAT DEFAULT 0.8,
    catch_radius FLOAT DEFAULT 1.5,
    reaction_time_ms FLOAT DEFAULT 200,
    coverage_skill FLOAT DEFAULT 0.7,
    closing_speed FLOAT DEFAULT 4.0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE formations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    external_id VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    slots JSONB NOT NULL,  -- [{role, position: {x, y}}]
    is_offensive BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE routes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    external_id VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    breakpoints JSONB NOT NULL,  -- [{x_yards, y_yards, time_ms}]
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE plays (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    external_id VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    formation_id UUID REFERENCES formations(id),
    assignments JSONB NOT NULL,  -- {role: route_id or null}
    qb_plan JSONB NOT NULL,  -- {progression_roles, max_time_to_throw_ms, scramble_allowed}
    tags VARCHAR(255)[],  -- ['redzone', 'goal_line', 'quick']
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE scenarios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    external_id VARCHAR(100),
    name VARCHAR(255) NOT NULL,
    defense_call JSONB NOT NULL,  -- {type, shell, rushers_count, rusher_position}
    defender_start_positions JSONB NOT NULL,  -- {D1: {x, y}, ...}
    situation_bucket VARCHAR(50),  -- 1ST_ANY, 3RD_SHORT, etc.
    tags VARCHAR(255)[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Simulation & Training
CREATE TABLE simulation_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    play_id UUID REFERENCES plays(id),
    scenario_ids UUID[],
    num_episodes INT NOT NULL,
    seed INT,
    mode VARCHAR(50),  -- EVAL, TRAIN
    metrics JSONB,  -- Full metrics object
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE trained_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    name VARCHAR(255),
    algorithm VARCHAR(50),  -- BANDIT, UCB
    config JSONB,  -- Training configuration
    policy_data BYTEA,  -- Pickled policy object
    metrics JSONB,  -- Training results
    play_ids UUID[],  -- Plays used in training
    scenario_ids UUID[],  -- Scenarios trained against
    created_at TIMESTAMP DEFAULT NOW()
);

-- Playbook Collections
CREATE TABLE playbooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID REFERENCES teams(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    play_ids UUID[],
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_players_team ON players(team_id);
CREATE INDEX idx_plays_team ON plays(team_id);
CREATE INDEX idx_scenarios_team ON scenarios(team_id);
CREATE INDEX idx_formations_team ON formations(team_id);
CREATE INDEX idx_routes_team ON routes(team_id);
```

### 1.3 SQLAlchemy Models

Create `fieldos_engine/db/models.py`:

```python
from sqlalchemy import Column, String, Float, Boolean, ForeignKey, ARRAY, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255))
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    teams = relationship("Team", back_populates="owner")

class Team(Base):
    __tablename__ = "teams"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    settings = Column(JSONB, default={})
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="teams")
    players = relationship("Player", back_populates="team")
    plays = relationship("Play", back_populates="team")

class Player(Base):
    __tablename__ = "players"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("teams.id"))
    external_id = Column(String(100))
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)
    speed = Column(Float, default=4.5)
    acceleration = Column(Float, default=2.0)
    route_sharpness = Column(Float, default=0.8)
    catch_radius = Column(Float, default=1.5)
    reaction_time_ms = Column(Float, default=200)
    coverage_skill = Column(Float, default=0.7)
    closing_speed = Column(Float, default=4.0)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)

    team = relationship("Team", back_populates="players")

# ... similar models for Formation, Route, Play, Scenario, etc.
```

---

## Phase 2: API Layer Updates

### 2.1 New CRUD Endpoints

Add to `fieldos_engine/api/routes.py`:

```python
# Players CRUD
@router.post("/teams/{team_id}/players", response_model=PlayerResponse)
@router.get("/teams/{team_id}/players", response_model=List[PlayerResponse])
@router.get("/teams/{team_id}/players/{player_id}", response_model=PlayerResponse)
@router.put("/teams/{team_id}/players/{player_id}", response_model=PlayerResponse)
@router.delete("/teams/{team_id}/players/{player_id}")

# Formations CRUD
@router.post("/teams/{team_id}/formations", response_model=FormationResponse)
@router.get("/teams/{team_id}/formations", response_model=List[FormationResponse])
@router.put("/teams/{team_id}/formations/{formation_id}", response_model=FormationResponse)
@router.delete("/teams/{team_id}/formations/{formation_id}")

# Routes CRUD
@router.post("/teams/{team_id}/routes", response_model=RouteResponse)
@router.get("/teams/{team_id}/routes", response_model=List[RouteResponse])
@router.put("/teams/{team_id}/routes/{route_id}", response_model=RouteResponse)
@router.delete("/teams/{team_id}/routes/{route_id}")

# Plays CRUD
@router.post("/teams/{team_id}/plays", response_model=PlayResponse)
@router.get("/teams/{team_id}/plays", response_model=List[PlayResponse])
@router.put("/teams/{team_id}/plays/{play_id}", response_model=PlayResponse)
@router.delete("/teams/{team_id}/plays/{play_id}")

# Scenarios CRUD
@router.post("/teams/{team_id}/scenarios", response_model=ScenarioResponse)
@router.get("/teams/{team_id}/scenarios", response_model=List[ScenarioResponse])
@router.put("/teams/{team_id}/scenarios/{scenario_id}", response_model=ScenarioResponse)
@router.delete("/teams/{team_id}/scenarios/{scenario_id}")

# Playbooks CRUD
@router.post("/teams/{team_id}/playbooks", response_model=PlaybookResponse)
@router.get("/teams/{team_id}/playbooks", response_model=List[PlaybookResponse])
@router.put("/teams/{team_id}/playbooks/{playbook_id}", response_model=PlaybookResponse)
@router.delete("/teams/{team_id}/playbooks/{playbook_id}")

# Training & Policies
@router.post("/teams/{team_id}/train", response_model=TrainingResponse)
@router.get("/teams/{team_id}/policies", response_model=List[PolicyResponse])
@router.get("/teams/{team_id}/policies/{policy_id}/recommend", response_model=RecommendationResponse)
```

### 2.2 Authentication Endpoints

```python
@router.post("/auth/register", response_model=UserResponse)
@router.post("/auth/login", response_model=TokenResponse)
@router.post("/auth/refresh", response_model=TokenResponse)
@router.get("/auth/me", response_model=UserResponse)
```

### 2.3 Database Session Dependency

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/fieldos")

engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

---

## Phase 3: Frontend Integration Points

### 3.1 Play Designer

Frontend components needed:
- **Formation Editor**: Drag-and-drop player positions on a field canvas
- **Route Builder**: Click-to-add breakpoints, adjust timing
- **Assignment Editor**: Assign routes to player slots
- **QB Plan Editor**: Set progression, timing, scramble rules

API endpoints used:
```
POST /teams/{team_id}/formations  -- Save new formation
POST /teams/{team_id}/routes      -- Save new route
POST /teams/{team_id}/plays       -- Save complete play
GET  /teams/{team_id}/plays/{id}/visualize  -- Generate preview GIF
```

### 3.2 Scenario Builder

Frontend components:
- **Defense Formation Editor**: Position D1-D5 on field
- **Coverage Selector**: Choose coverage type and shell
- **Rusher Configuration**: Toggle rush, select position (L/C/R)
- **Situation Tags**: Label for game situations

API endpoints:
```
POST /teams/{team_id}/scenarios
GET  /teams/{team_id}/scenarios/{id}/preview  -- Static diagram
```

### 3.3 Player Manager

Frontend components:
- **Player List**: View/filter all players
- **Player Card**: Edit attributes (speed, skills, etc.)
- **Roster Builder**: Assign players to roles

API endpoints:
```
GET  /teams/{team_id}/players
POST /teams/{team_id}/players
PUT  /teams/{team_id}/players/{id}
```

### 3.4 Simulation Dashboard

Frontend components:
- **Play Selector**: Choose plays to simulate
- **Scenario Selector**: Choose defensive looks
- **Results Table**: View metrics (completion %, yards, etc.)
- **Animation Player**: View GIF of simulations

API endpoints:
```
POST /teams/{team_id}/simulate
GET  /teams/{team_id}/simulations/{id}/trace  -- Get animation data
```

### 3.5 Training Center

Frontend components:
- **Training Configuration**: Set algorithm, steps, plays/scenarios
- **Progress Monitor**: Show training progress
- **Policy Browser**: View trained policies
- **Recommendation Display**: Get play calls for situations

API endpoints:
```
POST /teams/{team_id}/train
GET  /teams/{team_id}/policies
GET  /teams/{team_id}/policies/{id}/recommend?situation=3RD_SHORT
```

---

## Phase 4: Migration Strategy

### 4.1 Create Migration System

Use Alembic for database migrations:

```bash
pip install alembic
alembic init alembic
```

Configure `alembic/env.py` to use async engine.

### 4.2 Initial Migration

```python
# alembic/versions/001_initial.py
def upgrade():
    # Create all tables from schema above
    pass

def downgrade():
    # Drop all tables
    pass
```

### 4.3 Data Import from JSON

Create import script `scripts/import_json_to_db.py`:

```python
async def import_demo_data(db: AsyncSession, team_id: UUID):
    """Import existing JSON demo data into database."""

    # Load JSON files
    plays_data = load_json("fieldos_engine/data/demo/plays.json")
    scenarios_data = load_json("fieldos_engine/data/demo/scenarios.json")
    players_data = load_json("fieldos_engine/data/demo/players.json")

    # Import players
    for p in players_data:
        player = Player(
            team_id=team_id,
            external_id=p["id"],
            name=p["name"],
            role=p["role"],
            **p.get("offensive_attrs", {}),
            **p.get("defensive_attrs", {})
        )
        db.add(player)

    # Import plays, scenarios, etc.
    # ...

    await db.commit()
```

---

## Phase 5: Implementation Timeline

### Week 1-2: Database Foundation
- [ ] Set up PostgreSQL (local + cloud options)
- [ ] Create SQLAlchemy models
- [ ] Set up Alembic migrations
- [ ] Create initial migration
- [ ] Add database session dependency to FastAPI

### Week 3-4: CRUD Endpoints
- [ ] Implement Players CRUD
- [ ] Implement Formations CRUD
- [ ] Implement Routes CRUD
- [ ] Implement Plays CRUD
- [ ] Implement Scenarios CRUD
- [ ] Add request/response schemas

### Week 5: Authentication
- [ ] Add User model and auth endpoints
- [ ] Implement JWT authentication
- [ ] Add team membership logic
- [ ] Secure all endpoints with auth

### Week 6: Migration & Testing
- [ ] Create JSON import script
- [ ] Write integration tests
- [ ] Test with frontend prototype
- [ ] Document API with OpenAPI

---

## Configuration

### Environment Variables

```bash
# .env
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/fieldos
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional: Cloud database
# DATABASE_URL=postgresql+asyncpg://user:pass@your-cloud-host.com:5432/fieldos
```

### Dependencies to Add

```toml
# pyproject.toml additions
[project.optional-dependencies]
db = [
    "sqlalchemy>=2.0",
    "asyncpg>=0.29",
    "alembic>=1.13",
    "python-jose>=3.3",  # JWT
    "passlib>=1.7",      # Password hashing
    "bcrypt>=4.0",
]
```

---

## Summary

This plan provides a complete roadmap to:

1. **Store all game data in PostgreSQL** - Players, formations, routes, plays, scenarios
2. **Enable full CRUD operations** - Create, read, update, delete via REST API
3. **Support multi-tenancy** - Each team has isolated data
4. **Maintain compatibility** - Import existing JSON data
5. **Power a frontend** - Every entity is customizable via API

The frontend can then provide:
- Visual play designer with drag-and-drop
- Scenario builder for defensive looks
- Player attribute editor
- Simulation runner with live results
- Training dashboard for RL policies
- Playbook organization and sharing
