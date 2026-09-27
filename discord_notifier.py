import os
import requests
from datetime import datetime
from database import get_setting

def send_discord_notification_detailed(
    image_path: str,
    track_id: int,
    camera_name: str = "Entrance Camera",
    object_class: str = "person",
    confidence: float = 0.90,
    webhook_url: str = None
) -> tuple:
    """
    Sends a rich Discord notification with screenshot attachment and metadata embed.
    Returns (success: bool, detail_message: str).
    """
    if not webhook_url:
        webhook_url = get_setting("DISCORD_WEBHOOK_URL", os.getenv("DISCORD_WEBHOOK_URL", ""))
        
    if not webhook_url or not webhook_url.strip():
        msg = "Discord Webhook URL is not configured."
        print(f"ℹ️ {msg}")
        return False, msg

    webhook_url = webhook_url.strip()
    if not (webhook_url.startswith("http://") or webhook_url.startswith("https://")):
        msg = "Invalid Webhook URL format. Must start with http:// or https://"
        print(f"❌ {msg}")
        return False, msg

    if not os.path.exists(image_path):
        msg = f"Screenshot path does not exist: {image_path}"
        print(f"❌ {msg}")
        return False, msg

    filename = os.path.basename(image_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
                webhook_url,
                data={"payload_json": requests.compat.json.dumps(payload)},
                files={"file": (filename, img_file, "image/jpeg")},
                timeout=8
            )

        if response.status_code in [200, 204]:
            msg = f"Discord notification sent successfully for Track ID #{track_id}"
            print(f"✅ {msg}")
            return True, msg
        elif response.status_code in [401, 404]:
            err_json = {}
            try:
                err_json = response.json()
            except Exception:
                pass
            disc_msg = err_json.get("message", "")
            msg = f"Invalid Webhook Token or Webhook deleted ({response.status_code}): {disc_msg or 'Token not valid'}. Please re-create the webhook in Discord."
            print(f"❌ {msg}")
            return False, msg
        elif response.status_code == 429:
            msg = "Discord rate limit reached. Try again in a few seconds."
            print(f"⚠️ {msg}")
            return False, msg
        else:
            msg = f"Discord Webhook error ({response.status_code}): {response.text}"
            print(f"⚠️ {msg}")
            return False, msg
            
    except Exception as e:
        msg = f"Error sending Discord notification: {str(e)}"
        print(f"❌ {msg}")
        return False, msg

def send_discord_notification(
    image_path: str,
    track_id: int,
    camera_name: str = "Entrance Camera",
    object_class: str = "person",
    confidence: float = 0.90,
    webhook_url: str = None
) -> bool:
    """
    Wrapper for send_discord_notification_detailed returning simple boolean status.
    """
    success, _ = send_discord_notification_detailed(
        image_path=image_path,
        track_id=track_id,
        camera_name=camera_name,
        object_class=object_class,
        confidence=confidence,
        webhook_url=webhook_url
    )
    return success

