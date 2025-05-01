from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, DateTime, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from werkzeug.security import generate_password_hash, check_password_hash

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128))
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    records = relationship('OccupancyRecord', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class OccupancyRecord(Base):
    __tablename__ = 'occupancy_records'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    sitting_count = Column(Integer)
    total_chairs = Column(Integer)
    total_people = Column(Integer)
    image_path = Column(String)
    notes = Column(String)  # Yönetici notları için
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

# Veritabanı bağlantısı
engine = create_engine('sqlite:///library.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Admin kullanıcısı oluştur (ilk çalıştırmada)
def create_admin():
    session = Session()
    admin = session.query(User).filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@example.com',
            is_admin=True
        )
        admin.set_password('admin123')  # Varsayılan şifre
        session.add(admin)
        session.commit()
    session.close() 