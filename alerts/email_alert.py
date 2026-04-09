"""
email_alert.py — Email Notification System for AAPT
====================================================
Sends automated email alerts when:
  1. A student's engagement drifts (drops significantly)
  2. A teacher manually triggers an alert from the dashboard

We send TWO emails per alert:
  1. To the TEACHER: urgent notification with risk factors + intervention
  2. To the STUDENT: encouraging message with suggestions

Uses smtplib with SMTP_SSL (port 465) for secure Gmail connections.

Setup required:
  1. Enable 2-Factor Authentication on your Gmail account
  2. Generate an App Password at: https://myaccount.google.com/apppasswords
  3. Put the App Password in config.py → SMTP_PASSWORD
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import SMTP_HOST, SMTP_PORT, SMTP_SENDER, SMTP_PASSWORD


def send_drift_alert(student_name, student_email, teacher_email,
                     z_score, top_factors, smtp_config=None):
    """
    Send alert emails to both teacher and student.
    
    Parameters:
      student_name: student's display name
      student_email: student's email address
      teacher_email: teacher's email address
      z_score: the drift Z-score value
      top_factors: list of dicts with 'feature', 'shap_value', 'suggestion'
      smtp_config: optional dict to override default SMTP settings
                   keys: host, port, sender_email, password
    
    Returns:
      dict with:
        - success: bool
        - message: status message
        - teacher_sent: bool
        - student_sent: bool
    """
    # Use provided config or fall back to defaults from config.py
    if smtp_config is None:
        smtp_config = {
            "host": SMTP_HOST,
            "port": SMTP_PORT,
            "sender_email": SMTP_SENDER,
            "password": SMTP_PASSWORD,
        }
    
    # Check if SMTP is configured (not placeholder values)
    if smtp_config["password"] == "your_app_password_here":
        return {
            "success": False,
            "message": "SMTP not configured. Update SMTP_PASSWORD in config.py.",
            "teacher_sent": False,
            "student_sent": False,
        }
    
    # Build the risk factors text for the email body
    factors_text = ""
    for i, factor in enumerate(top_factors, 1):
        fname = factor.get("feature", "Unknown")
        suggestion = factor.get("suggestion", "")
        factors_text += f"  {i}. {fname}: {suggestion}\n"
    
    if not factors_text:
        factors_text = "  No specific factors identified.\n"
    
    teacher_sent = False
    student_sent = False
    errors = []
    
    try:
        # ── Connect to Gmail SMTP server ────────────────────
        # SMTP_SSL uses SSL encryption from the start (port 465)
        # This is more secure than STARTTLS (port 587)
        server = smtplib.SMTP_SSL(smtp_config["host"], smtp_config["port"])
        server.login(smtp_config["sender_email"], smtp_config["password"])
        
        # ── Email 1: Alert to Teacher ───────────────────────
        if teacher_email:
            teacher_msg = MIMEMultipart("alternative")
            teacher_msg["Subject"] = f"AAPT Alert: {student_name} needs attention"
            teacher_msg["From"] = smtp_config["sender_email"]
            teacher_msg["To"] = teacher_email
            
            # Plain text version
            teacher_body_text = f"""
AAPT — Adaptive Academic Performance Tracking System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  ALERT: Student Engagement Drift Detected

Student: {student_name}
Z-Score: {z_score:.2f} (threshold: ±2.0)

This student's engagement has dropped significantly compared to
their own baseline. Immediate attention may be required.

Top Risk Factors & Recommended Actions:
{factors_text}

Suggested Next Steps:
  • Schedule a one-on-one meeting with the student
  • Check if the student is facing personal difficulties
  • Consider assigning a peer mentor
  • Review the student's progress in the AAPT dashboard

—
This alert was generated automatically by the AAPT System.
            """
            
            # HTML version (for better formatting in email clients)
            teacher_body_html = f"""
<html>
<body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #f8f9fa; padding: 20px;">
  <div style="background: #1a1a2e; color: white; padding: 20px 30px; border-radius: 12px 12px 0 0;">
    <h2 style="margin: 0;">📊 AAPT Alert</h2>
    <p style="margin: 5px 0 0; opacity: 0.8;">Adaptive Academic Performance Tracking</p>
  </div>
  <div style="background: white; padding: 25px 30px; border-radius: 0 0 12px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; border-radius: 4px; margin-bottom: 20px;">
      <strong>⚠️ Engagement Drift Detected</strong>
    </div>
    <p><strong>Student:</strong> {student_name}</p>
    <p><strong>Z-Score:</strong> <span style="color: #dc3545; font-weight: bold;">{z_score:.2f}</span> (threshold: ±2.0)</p>
    <p>This student's engagement has dropped significantly compared to their baseline.</p>
    <h3 style="color: #1a1a2e;">Top Risk Factors</h3>
    <ul>
"""
            for factor in top_factors:
                teacher_body_html += f'      <li><strong>{factor.get("feature", "")}:</strong> {factor.get("suggestion", "")}</li>\n'
            
            teacher_body_html += """
    </ul>
    <h3 style="color: #1a1a2e;">Suggested Actions</h3>
    <ul>
      <li>Schedule a one-on-one meeting</li>
      <li>Check for personal difficulties</li>
      <li>Consider assigning a peer mentor</li>
    </ul>
    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
    <p style="color: #888; font-size: 12px;">Auto-generated by AAPT System</p>
  </div>
</body>
</html>
            """
            
            teacher_msg.attach(MIMEText(teacher_body_text, "plain"))
            teacher_msg.attach(MIMEText(teacher_body_html, "html"))
            
            try:
                server.sendmail(
                    smtp_config["sender_email"],
                    teacher_email,
                    teacher_msg.as_string()
                )
                teacher_sent = True
            except Exception as e:
                errors.append(f"Teacher email failed: {str(e)}")
        
        # ── Email 2: Encouragement to Student ──────────────
        if student_email:
            student_msg = MIMEMultipart("alternative")
            student_msg["Subject"] = "Your learning activity this week — AAPT"
            student_msg["From"] = smtp_config["sender_email"]
            student_msg["To"] = student_email
            
            # Get the top suggestion for the student
            top_suggestion = ""
            if top_factors:
                top_suggestion = top_factors[0].get("suggestion", "Keep engaging with your course materials!")
            
            student_body_text = f"""
Hi {student_name},

We noticed your activity on the learning platform has been different 
this week. Don't worry — we're here to help!

💡 Tip: {top_suggestion}

Remember:
  • Every small step counts toward your success
  • Your teachers are ready to support you
  • It's okay to ask for help when you need it

Keep going — you've got this! 💪

—
AAPT — Adaptive Academic Performance Tracking System
            """
            
            student_body_html = f"""
<html>
<body style="font-family: 'Segoe UI', Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
  <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px 30px; border-radius: 12px 12px 0 0;">
    <h2 style="margin: 0;">Hey {student_name}! 👋</h2>
    <p style="margin: 5px 0 0; opacity: 0.9;">Your weekly activity update</p>
  </div>
  <div style="background: white; padding: 25px 30px; border-radius: 0 0 12px 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <p>We noticed your activity has been a bit different this week. That's completely okay!</p>
    <div style="background: #e8f4fd; border-left: 4px solid #2196F3; padding: 12px 16px; border-radius: 4px; margin: 15px 0;">
      <strong>💡 Suggestion:</strong> {top_suggestion}
    </div>
    <h3>Remember:</h3>
    <ul>
      <li>Every small step counts 🎯</li>
      <li>Your teachers are here to help 🤝</li>
      <li>It's okay to ask for support 💬</li>
    </ul>
    <p style="font-size: 18px; text-align: center; margin-top: 20px;">Keep going — you've got this! 💪</p>
    <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
    <p style="color: #888; font-size: 12px;">AAPT — Adaptive Academic Performance Tracking System</p>
  </div>
</body>
</html>
            """
            
            student_msg.attach(MIMEText(student_body_text, "plain"))
            student_msg.attach(MIMEText(student_body_html, "html"))
            
            try:
                server.sendmail(
                    smtp_config["sender_email"],
                    student_email,
                    student_msg.as_string()
                )
                student_sent = True
            except Exception as e:
                errors.append(f"Student email failed: {str(e)}")
        
        # Close the SMTP connection
        server.quit()
        
    except smtplib.SMTPAuthenticationError:
        return {
            "success": False,
            "message": "SMTP authentication failed. Check your email and app password in config.py.",
            "teacher_sent": False,
            "student_sent": False,
        }
    except smtplib.SMTPConnectError:
        return {
            "success": False,
            "message": "Could not connect to SMTP server. Check your internet connection.",
            "teacher_sent": False,
            "student_sent": False,
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Email sending failed: {str(e)}",
            "teacher_sent": teacher_sent,
            "student_sent": student_sent,
        }
    
    # Build result
    if teacher_sent and student_sent:
        msg = "Both emails sent successfully!"
    elif teacher_sent:
        msg = "Teacher email sent. " + (errors[0] if errors else "Student email not sent.")
    elif student_sent:
        msg = "Student email sent. " + (errors[0] if errors else "Teacher email not sent.")
    else:
        msg = "No emails sent. " + " | ".join(errors)
    
    return {
        "success": teacher_sent or student_sent,
        "message": msg,
        "teacher_sent": teacher_sent,
        "student_sent": student_sent,
    }
