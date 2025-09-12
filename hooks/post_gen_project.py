"""
Cookiecutter hook для автоматического создания репозитория в GitHub
после генерации проекта из шаблона.
"""

import os
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    # Если requests не установлен, установим его
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


def create_github_repo(repo_name, description="", private=False, github_token=None):
    """
    Создает новый репозиторий в GitHub.
    
    Args:
        repo_name (str): Название репозитория
        description (str): Описание репозитория
        private (bool): Приватный ли репозиторий
        github_token (str): Токен доступа к GitHub API
        
    Returns:
        str: URL созданного репозитория или None в случае ошибки
    """
    if not github_token:
        print("Токен GitHub не предоставлен. Пропускаем создание репозитория.")
        return None
    
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "name": repo_name,
        "description": description,
        "private": private,
        "auto_init": False  # Не инициализировать с README
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        repo_info = response.json()
        return repo_info.get("html_url")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при создании репозитория в GitHub: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Ответ сервера: {e.response.text}")
        return None


def initialize_git_and_push(repo_url, github_token=None):
    """
    Инициализирует git репозиторий, делает первый коммит и пушит в удаленный репозиторий.
    
    Args:
        repo_url (str): URL удаленного репозитория
        github_token (str): Токен доступа к GitHub API
    """
    try:
        # Инициализируем git репозиторий
        subprocess.run(["git", "init"], check=True)
        
        # Добавляем все файлы
        subprocess.run(["git", "add", "."], check=True)
        
        # Делаем первый коммит
        subprocess.run(["git", "commit", "-m", "Initial commit from cookiecutter template"], check=True)
        
        # Добавляем удаленный репозиторий
        if github_token:
            # Используем токен в URL для аутентификации
            auth_repo_url = repo_url.replace("https://", f"https://{github_token}@")
            subprocess.run(["git", "remote", "add", "origin", auth_repo_url], check=True)
        else:
            subprocess.run(["git", "remote", "add", "origin", repo_url], check=True)
        
        # Пушим изменения
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        
        print("Репозиторий успешно инициализирован и код загружен в GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при работе с git: {e}")
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")


def main():
    """Основная функция хука."""
    # Получаем переменные из cookiecutter контекста
    project_name = "{{ cookiecutter.project_name }}"
    repo_name = "{{ cookiecutter.repo_name }}"
    description = "{{ cookiecutter.description }}"
    
    print(f"Настройка проекта '{project_name}'...")
    
    # Переходим в директорию проекта
    project_dir = Path(repo_name)
    if project_dir.exists():
        os.chdir(project_dir)
    else:
        print(f"Директория проекта {project_dir} не найдена!")
        return
    
    # Спрашиваем пользователя, хочет ли он создать репозиторий в GitHub
    create_repo = input("Хотите создать репозиторий в GitHub? (y/N): ").strip().lower()
    if create_repo not in ['y', 'yes', 'д', 'да']:
        print("Создание репозитория в GitHub пропущено.")
        return
    
    # Запрашиваем токен GitHub
    github_token = input("Введите ваш GitHub токен (или оставьте пустым для пропуска): ").strip()
    if not github_token:
        print("Токен GitHub не предоставлен. Пропускаем создание репозитория.")
        return
    
    # Спрашиваем, должен ли репозиторий быть приватным
    private_repo = input("Сделать репозиторий приватным? (y/N): ").strip().lower()
    is_private = private_repo in ['y', 'yes', 'д', 'да']
    
    # Создаем репозиторий в GitHub
    print("Создание репозитория в GitHub...")
    repo_url = create_github_repo(
        repo_name=repo_name,
        description=description,
        private=is_private,
        github_token=github_token
    )
    
    if repo_url:
        print(f"Репозиторий успешно создан: {repo_url}")
        # Инициализируем git и пушим код
        initialize_git_and_push(repo_url, github_token)
    else:
        print("Не удалось создать репозиторий в GitHub.")


if __name__ == "__main__":
    main()
