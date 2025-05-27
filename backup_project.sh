#!/bin/bash

# Backup BTC Price Predictor project

echo "Backing up BTC Price Predictor project..."

# Get current date and time for backup filename
BACKUP_DATE=$(date +"%Y%m%d_%H%M%S")
BACKUP_DIR="backups"
BACKUP_NAME="btc_predictor_backup_${BACKUP_DATE}"

# Create backup directory if it doesn't exist
mkdir -p $BACKUP_DIR

# Function to confirm backup
confirm() {
    read -p "$1 (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Ask what to backup
echo "What would you like to backup?"
echo "1. Code only (src, tests, config)"
echo "2. Code and data (src, tests, config, data)"
echo "3. Full project (all files except virtual environments and node_modules)"
echo "4. Custom selection"
read -p "Enter option (1-4): " backup_option

case $backup_option in
    1)
        echo "Backing up code only..."
        BACKUP_ITEMS="src tests config main.py requirements.txt README.md"
        ;;
    2)
        echo "Backing up code and data..."
        BACKUP_ITEMS="src tests config data main.py requirements.txt README.md"
        ;;
    3)
        echo "Backing up full project..."
        BACKUP_ITEMS="--exclude='venv' --exclude='env' --exclude='node_modules' --exclude='__pycache__' --exclude='.git' ."
        ;;
    4)
        echo "Enter directories and files to backup (space-separated):"
        read -p "> " BACKUP_ITEMS
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Ask for backup format
echo "Select backup format:"
echo "1. tar.gz (compressed tar archive)"
echo "2. zip (zip archive)"
read -p "Enter option (1-2): " format_option

case $format_option in
    1)
        BACKUP_FORMAT="tar.gz"
        BACKUP_FILE="${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
        
        echo "Creating tar.gz backup..."
        if [ "$backup_option" -eq 3 ]; then
            tar -czf "$BACKUP_FILE" $BACKUP_ITEMS
        else
            tar -czf "$BACKUP_FILE" $BACKUP_ITEMS
        fi
        ;;
    2)
        BACKUP_FORMAT="zip"
        BACKUP_FILE="${BACKUP_DIR}/${BACKUP_NAME}.zip"
        
        echo "Creating zip backup..."
        if [ "$backup_option" -eq 3 ]; then
            zip -r "$BACKUP_FILE" . -x "venv/*" "env/*" "node_modules/*" "__pycache__/*" ".git/*"
        else
            zip -r "$BACKUP_FILE" $BACKUP_ITEMS
        fi
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: $BACKUP_FILE"
    echo "Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    echo "❌ Backup failed"
    exit 1
fi

# Ask if user wants to upload backup to cloud storage
if confirm "Do you want to upload the backup to cloud storage?"; then
    echo "Select cloud storage provider:"
    echo "1. AWS S3"
    echo "2. Google Cloud Storage"
    echo "3. Azure Blob Storage"
    echo "4. Skip upload"
    read -p "Enter option (1-4): " cloud_option
    
    case $cloud_option in
        1)
            echo "AWS S3 selected."
            if command -v aws >/dev/null 2>&1; then
                read -p "Enter S3 bucket name: " s3_bucket
                echo "Uploading to AWS S3..."
                aws s3 cp "$BACKUP_FILE" "s3://${s3_bucket}/${BACKUP_NAME}.${BACKUP_FORMAT}"
                
                if [ $? -eq 0 ]; then
                    echo "✅ Backup uploaded to S3: s3://${s3_bucket}/${BACKUP_NAME}.${BACKUP_FORMAT}"
                else
                    echo "❌ S3 upload failed"
                fi
            else
                echo "❌ AWS CLI not found. Please install it first."
            fi
            ;;
        2)
            echo "Google Cloud Storage selected."
            if command -v gsutil >/dev/null 2>&1; then
                read -p "Enter GCS bucket name: " gcs_bucket
                echo "Uploading to Google Cloud Storage..."
                gsutil cp "$BACKUP_FILE" "gs://${gcs_bucket}/${BACKUP_NAME}.${BACKUP_FORMAT}"
                
                if [ $? -eq 0 ]; then
                    echo "✅ Backup uploaded to GCS: gs://${gcs_bucket}/${BACKUP_NAME}.${BACKUP_FORMAT}"
                else
                    echo "❌ GCS upload failed"
                fi
            else
                echo "❌ gsutil not found. Please install Google Cloud SDK first."
            fi
            ;;
        3)
            echo "Azure Blob Storage selected."
            if command -v az >/dev/null 2>&1; then
                read -p "Enter Azure storage account name: " az_account
                read -p "Enter Azure container name: " az_container
                echo "Uploading to Azure Blob Storage..."
                az storage blob upload --account-name "$az_account" --container-name "$az_container" --name "${BACKUP_NAME}.${BACKUP_FORMAT}" --file "$BACKUP_FILE"
                
                if [ $? -eq 0 ]; then
                    echo "✅ Backup uploaded to Azure: ${az_account}/${az_container}/${BACKUP_NAME}.${BACKUP_FORMAT}"
                else
                    echo "❌ Azure upload failed"
                fi
            else
                echo "❌ Azure CLI not found. Please install it first."
            fi
            ;;
        4)
            echo "Skipping cloud upload."
            ;;
        *)
            echo "Invalid option. Skipping cloud upload."
            ;;
    esac
fi

# Clean up old backups
if confirm "Do you want to clean up old backups?"; then
    read -p "Keep how many recent backups? " keep_count
    
    # List backups sorted by modification time
    backup_count=$(ls -1 "${BACKUP_DIR}"/* | wc -l)
    
    if [ "$backup_count" -gt "$keep_count" ]; then
        echo "Removing old backups..."
        ls -1t "${BACKUP_DIR}"/* | tail -n +$((keep_count+1)) | xargs rm -f
        echo "✅ Old backups removed"
    else
        echo "No old backups to remove"
    fi
fi

echo "Backup process completed!"