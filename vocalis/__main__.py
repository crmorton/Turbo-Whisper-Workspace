"""
Main entry point for Vocalis package

This module provides command-line interfaces for:
- Running the FastAPI server
- Running the Gradio UI
- Running the security monitor
"""

import argparse
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vocalis")

def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server"""
    try:
        import uvicorn
        from vocalis.api.main import app
        
        logger.info(f"Starting Vocalis API server on {host}:{port}")
        uvicorn.run("vocalis.api.main:app", host=host, port=port, reload=reload)
    except ImportError:
        logger.error("Failed to import required packages. Please install with 'pip install vocalis[api]'")
        sys.exit(1)

def run_ui(share: bool = False):
    """Run the Gradio UI"""
    try:
        import gradio as gr
        
        # Import the app from the UI module
        from vocalis.ui.app import demo
        
        logger.info("Starting Vocalis UI")
        demo.launch(share=share)
    except ImportError:
        logger.error("Failed to import required packages. Please install with 'pip install vocalis[ui]'")
        sys.exit(1)

def run_security_monitor(input_path: str, output_dir: str = "security_incidents", 
                        min_threat_level: int = 2, bar_specific: bool = False):
    """Run the security monitor on a file or directory"""
    try:
        if bar_specific:
            from vocalis.security.bar_security_monitor import BarSecurityMonitor, monitor_bar_directory
            
            if os.path.isdir(input_path):
                logger.info(f"Monitoring bar directory: {input_path}")
                monitor_bar_directory(input_path, output_dir, min_threat_level)
            elif os.path.isfile(input_path):
                logger.info(f"Processing bar audio file: {input_path}")
                monitor = BarSecurityMonitor(output_dir=output_dir, min_threat_level=min_threat_level)
                incident = monitor.process_audio_file(input_path)
                
                if incident:
                    logger.warning("⚠️ Bar security incident detected!")
                    print(str(incident))
                else:
                    logger.info("No bar security concerns detected")
            else:
                logger.error(f"Input not found: {input_path}")
                sys.exit(1)
        else:
            from vocalis.security.security_monitor import SecurityMonitor, monitor_directory
            
            if os.path.isdir(input_path):
                logger.info(f"Monitoring directory: {input_path}")
                monitor_directory(input_path, output_dir, min_threat_level)
            elif os.path.isfile(input_path):
                logger.info(f"Processing audio file: {input_path}")
                monitor = SecurityMonitor(output_dir=output_dir, min_threat_level=min_threat_level)
                incident = monitor.process_audio_file(input_path)
                
                if incident:
                    logger.warning("⚠️ Security incident detected!")
                    print(str(incident))
                else:
                    logger.info("No security concerns detected")
            else:
                logger.error(f"Input not found: {input_path}")
                sys.exit(1)
    except ImportError:
        logger.error("Failed to import required packages. Please install with 'pip install vocalis[security]'")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vocalis - Advanced Audio Processing")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Run the FastAPI server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    api_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Run the Gradio UI")
    ui_parser.add_argument("--share", action="store_true", help="Create a public link")
    
    # Security monitor command
    security_parser = subparsers.add_parser("security", help="Run the security monitor")
    security_parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    security_parser.add_argument("--output", "-o", default="security_incidents", help="Output directory for incident reports")
    security_parser.add_argument("--threat-level", "-t", type=int, default=2, help="Minimum threat level to report (1-5)")
    security_parser.add_argument("--bar", "-b", action="store_true", help="Use bar-specific security monitoring")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "api":
        run_api_server(args.host, args.port, args.reload)
    elif args.command == "ui":
        run_ui(args.share)
    elif args.command == "security":
        run_security_monitor(args.input, args.output, args.threat_level, args.bar)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()