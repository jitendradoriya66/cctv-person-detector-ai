import os
import requests
from datetime import datetime
from database import get_setting

def send_discord_notification(
    image_path: str,
    track_id: int,
    camera_name: str = "Entrance Camera",
    object_class: str = "person",
    confidence: float = 0.90,
    webhook_url: str = None
) -> bool:
    """
    Sends a rich Discord notification with screenshot attachment and metadata embed.
    """
    if not webhook_url:
        webhook_url = get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
        
    if not webhook_url or not webhook_url.strip():
        print("ℹ️ Discord Webhook URL not set. Skipping notification.")
        return False

    if not os.path.exists(image_path):
        print(f"❌ Screenshot path does not exist: {image_path}")
        return False

    filename = os.path.basename(image_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Construct rich embed payload
    payload = {
        "username": "AI CCTV Monitor",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/3067/3067256.png",
        "embeds": [
            {
                "title": f"🚨 {object_class.capitalize()} Detected!",
                "description": f"A **{object_class}** was detected on camera stream.",
                "color": 15158332,  # Crimson Red
                "fields": [
                    {
                        "name": "📹 Camera",
                        "value": camera_name,
                        "inline": True
                    },
                    {
                        "name": "🆔 Track ID",
                        "value": f"#{track_id}",
                        "inline": True
                    },
                    {
                        "name": "🎯 Confidence",
                        "value": f"{int(confidence * 100)}%",
                        "inline": True
                    },
                    {
                        "name": "🕒 Time",
                        "value": now_str,
                        "inline": False
                    }
                ],
                "image": {
                    "url": f"attachment://{filename}"
                },
                "footer": {
                    "text": "AI CCTV Security System • Real-Time Protection"
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        ]
    }

    try:
        with open(image_path, "rb") as img_file:
            response = requests.post(
                webhook_url.strip(),
                data={"payload_json": requests.compat.json.dumps(payload)},
                files={"file": (filename, img_file, "image/jpeg")},
                timeout=8
            )

        if response.status_code in [200, 204]:
            print(f"✅ Discord notification sent successfully for Track ID {track_id}")
            return True
        else:
            print(f"⚠️ Discord Webhook response error ({response.status_code}): {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending Discord notification: {e}")
        return False
