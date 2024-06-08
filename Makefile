# Makefile

# ターゲット名
.PHONY: build

# ビルドタスク
build:
	docker compose build

run:
	docker compose run --rm icefall

# コンテナを停止する
down:
	docker compose down

# ログを表示する
logs:
	docker compose logs -f

# クリーンアップ
clean:
	docker compose down --rmi all --volumes --remove-orphans
	rm -f tools