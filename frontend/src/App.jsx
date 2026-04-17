import { useCallback, useEffect, useState } from 'react'
import './App.css'

/** Vite 开发在 5173，相对路径 /recommend 会打到 Vite；默认指向本机 FastAPI */
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:8000'

function App() {
  /** 用字符串存输入框，避免清空时 Number('')===0 导致删不掉、前面多 0 */
  const [userId, setUserId] = useState('1')
  const [topN, setTopN] = useState('5')
  const [items, setItems] = useState([])
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  /** 是否已完成过至少一次「成功拿到 JSON」的请求（与 static 里 render 后 empty 文案一致） */
  const [listSettled, setListSettled] = useState(false)

  const showErr = (msg) => {
    setError(msg || '')
  }

  const applyResponseBody = (body) => {
    const list = body.items || []
    setItems(list)
    setListSettled(true)
  }

  const fetchRecommend = useCallback(async () => {
    showErr('')
    const uid = parseInt(String(userId), 10)
    const tn = parseInt(String(topN), 10)

    if (Number.isNaN(uid) || uid < 1) {
      showErr('请输入有效的用户 ID（正整数）。')
      return
    }
    if (Number.isNaN(tn) || tn < 1 || tn > 50) {
      showErr('top_n 需在 1～50 之间。')
      return
    }

    setLoading(true)
    try {
      const url = `${API_BASE}/recommend/${encodeURIComponent(uid)}?top_n=${encodeURIComponent(tn)}`
      const res = await fetch(url)
      const body = await res.json().catch(() => ({}))
      if (!res.ok) {
        showErr(
          body.detail != null ? String(body.detail) : `请求失败 (${res.status})`,
        )
        return
      }
      applyResponseBody(body)
    } catch {
      showErr('网络错误：请确认已启动服务（python -m uvicorn app:app）。')
    } finally {
      setLoading(false)
    }
  }, [userId, topN])

  // 与 static/index.html 一致：进入页面自动请求一次；之后仅「获取推荐」按钮触发
  useEffect(() => {
    void fetchRecommend()
    // eslint-disable-next-line react-hooks/exhaustive-deps -- 只挂载跑一次，不随 userId/topN 自动重拉
  }, [])

  const docsHref = `${API_BASE}/docs`

  return (
    <div className="wrap">
      <h1>电影推荐</h1>
      <p className="sub">
        基于用户协同过滤（MovieLens 数据）。输入用户 ID 与返回条数后获取推荐。
      </p>

      <div className="card">
        <div className="row">
          <div>
            <label htmlFor="userId">用户 ID</label>
            <input
              id="userId"
              type="number"
              min={1}
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="topN">返回条数 top_n</label>
            <input
              id="topN"
              type="number"
              min={1}
              max={50}
              value={topN}
              onChange={(e) => setTopN(e.target.value)}
            />
          </div>
        </div>
        <button type="button" disabled={loading} onClick={fetchRecommend}>
          获取推荐
        </button>
        {error ? <p className="err show">{error}</p> : <p className="err" />}
      </div>

      <div className="card">
        <label>推荐结果</label>
        <ul className="list">
          {items.map((item) => (
            <li key={item.movie_id}>
              <span className="title">
                {item.title != null
                  ? item.title
                  : `(无标题) #${item.movie_id}`}
              </span>
              <span className="score">
                {typeof item.score === 'number'
                  ? item.score.toFixed(4)
                  : String(item.score)}
              </span>
            </li>
          ))}
        </ul>
        {items.length === 0 && loading && (
          <p className="empty">加载中…</p>   /*这里的逻辑是，第一次进页面或上一次空结果时显示加载中，有过推荐结果时 即显示旧页面时不显示加载中*/
        )} 
        {items.length === 0 && listSettled && !loading && (
          <p className="empty">该条件下无推荐结果（或列表为空）。</p>
        )}
      </div>

      <footer>
        API 文档：
        <a href={docsHref} target="_blank" rel="noopener noreferrer">
          /docs
        </a>
        · React 版（Vite）；后端默认{' '}
        <code>{API_BASE}</code>
      </footer>
    </div>
  )
}

export default App
/*daily work*/