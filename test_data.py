from mongodb_database import check_connection, create_indexes

if check_connection():
    print("MongoDB connected successfully!")

    create_indexes()
    print("MongoDB indexes created successfully.")
else:
    print("MongoDB connection failed.")