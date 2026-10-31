import { CompetitiveQuestion, CompetitiveExamsProgress, UserAttemptRecord } from '../types/competitive';
import questionsData from '../data/competitiveQuestions.json';

const STORAGE_KEY = 'competitive_exams_progress_v1';

export const competitiveService = {
  getAllQuestions(): CompetitiveQuestion[] {
    return questionsData as CompetitiveQuestion[];
  },

  getTopics(): string[] {
    const questions = this.getAllQuestions();
    const set = new Set<string>();
    questions.forEach(q => set.add(q.topic));
    return Array.from(set);
  },

  getProgress(): CompetitiveExamsProgress {
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (raw) {
          return JSON.parse(raw);
        }
      }
    } catch (e) {
      // Safe fallback
    }
    return { attempts: {}, bookmarks: [] };
  },

  saveAttempt(questionId: string, selectedIndex: number, isCorrect: boolean): CompetitiveExamsProgress {
    const progress = this.getProgress();
    progress.attempts[questionId] = {
      questionId,
      selectedIndex,
      isCorrect,
      timestamp: new Date().toISOString()
    };
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
      }
    } catch (e) {
      // Safe fallback
    }
    return progress;
  },

  toggleBookmark(questionId: string): CompetitiveExamsProgress {
    const progress = this.getProgress();
    const idx = progress.bookmarks.indexOf(questionId);
    if (idx >= 0) {
      progress.bookmarks.splice(idx, 1);
    } else {
      progress.bookmarks.push(questionId);
    }
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
      }
    } catch (e) {
      // Safe fallback
    }
    return progress;
  },

  resetProgress(): CompetitiveExamsProgress {
    const clean: CompetitiveExamsProgress = { attempts: {}, bookmarks: [] };
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(clean));
      }
    } catch (e) {
      // Safe fallback
    }
    return clean;
  }
};
