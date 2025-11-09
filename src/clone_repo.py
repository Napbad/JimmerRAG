import os
import subprocess
from typing import Optional
from pathlib import Path

def clone_git_repository(
        repo_url: str,
        target_dir: Path,
        branch: Optional[str] = None,
        commit: Optional[str] = None,
        depth: Optional[int] = None
) -> bool:
    """
    Clone Git repository to specified directory

    Args:
        repo_url: Git repository URL
        target_dir: Target directory path
        branch: Branch name (optional)
        commit: Commit hash value (optional)
        depth: Shallow clone depth (optional)

    Returns:
        bool: Whether the operation was successful
    """

    try:
        # If target directory exists and is not empty, clean it first
        if os.path.exists(target_dir.__str__()) and os.listdir(target_dir.__str__()):
            print(f"Target directory {target_dir.__str__()} already exists and is not empty")
            return False

        # Build git clone command
        cmd = ["git", "clone"]

        # Add branch parameter
        if branch:
            cmd.extend(["-b", branch])

        # Add shallow clone parameter
        if depth:
            cmd.extend(["--depth", str(depth)])

        # Add repository URL and target directory
        cmd.extend([repo_url, target_dir.__str__()])

        # Execute clone operation
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Clone failed: {result.stderr}")
            return False

        # If specific commit is specified, checkout to that commit
        if commit:
            # Switch to target directory
            original_dir = os.getcwd()
            os.chdir(target_dir.__str__())

            try:
                # Execute checkout operation
                checkout_result = subprocess.run(
                    ["git", "checkout", commit],
                    capture_output=True,
                    text=True
                )
                if checkout_result.returncode != 0:
                    print(f"Checkout commit failed: {checkout_result.stderr}")
                    return False
            finally:
                # Restore original working directory
                os.chdir(original_dir)

        print(f"Successfully cloned code to {target_dir.__str__()}")
        return True

    except Exception as e:
        print(f"Error occurred during operation: {str(e)}")
        return False
