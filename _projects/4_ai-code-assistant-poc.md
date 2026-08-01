---
layout: page
title: 생성형 AI 기반 개발자 생산성 향상 AI Code Assistant PoC
description: 사내 코드베이스로 파인튜닝한 CodeLlama 기반 코드 어시스턴트 효용성 검증 PoC
img:
importance: 4
category: work
---

**기간**: 2024.02 · **소속**: 현대오토에버 인공지능기술팀

## 배경

생성형 AI 활용에 따른 기능별 impact 조사에서 Software Engineering 부문이 가장 높은 것으로 나타나, 사내 개발자를 대상으로 Code Assistant의 효용성과 생산성 향상 여부를 직접 검증해보기로 했습니다.

## 개요

사내 이커머스 웹 애플리케이션 개발 조직 소속 개발자 11명을 대상으로, CodeLlama의 모델 크기별 생산성 향상 정도와 소량의 기 개발 코드로 미세조정했을 때 전/후 생산성 향상 여부를 파악하는 것을 목표로 했습니다.

## 개발 내용

- 사내 이커머스 웹 애플리케이션 소스코드(Java Spring 4M 토큰, React 2.9M 토큰)를 학습 데이터로 CodeLlama를 QLoRA + Instruct tuning 방식으로 미세조정했습니다.
- Ollama 기반 API 서버를 구축했습니다.
- VS Code와 IntelliJ 개발자 도구 내 Continue 확장을 통해 개발자들이 바로 사용해볼 수 있는 테스트 환경을 제공했습니다.
