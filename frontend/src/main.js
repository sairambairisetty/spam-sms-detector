import './style.css'

const messageInput = document.getElementById('messageInput')
const checkBtn = document.getElementById('checkBtn')
const resultBox = document.getElementById('resultBox')

checkBtn.addEventListener('click', async () => {
  const message = messageInput.value.trim()
  if (!message) return

  checkBtn.disabled = true
  checkBtn.innerText = 'Analyzing...'
  resultBox.className = 'result-box hidden'

  try {
    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: message }),
    })

    const data = await response.json()
    const resultText = data.result || data.prediction

    resultBox.innerText = resultText

    if (resultText.includes('SPAM')) {
      resultBox.className = 'result-box spam'
    } else if (resultText.includes('HAM')) {
      resultBox.className = 'result-box ham'
    } else {
      resultBox.className = 'result-box error'
    }
  } catch (error) {
    console.error(error)
    resultBox.innerText = 'Error: Backend is not reachable'
    resultBox.className = 'result-box error'
  } finally {
    checkBtn.disabled = false
    checkBtn.innerText = 'Check Message'
  }
})