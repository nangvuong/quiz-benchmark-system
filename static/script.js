document.getElementById('generate-btn').addEventListener('click', async () => {
    const context = document.getElementById('context').value.trim();
    if (!context) {
        alert("Please enter some text context.");
        return;
    }

    const checkboxes = document.querySelectorAll('.checkbox-group input[type="checkbox"]:checked');
    const models = Array.from(checkboxes).map(cb => cb.value);

    if (models.length === 0) {
        alert("Please select at least one model.");
        return;
    }

    const btn = document.getElementById('generate-btn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.loader');
    const resultsSection = document.getElementById('results-section');

    // UI Loading state
    btn.disabled = true;
    btnText.textContent = "Generating...";
    loader.classList.remove('hidden');
    resultsSection.innerHTML = '';

    try {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ context, models })
        });

        const data = await response.json();

        if (response.ok) {
            renderResults(data.results);
        } else {
            alert(data.error || "An error occurred.");
        }
    } catch (err) {
        alert("Network error. Make sure the backend is running.");
        console.error(err);
    } finally {
        btn.disabled = false;
        btnText.textContent = "Generate MCQs";
        loader.classList.add('hidden');
    }
});

function renderResults(resultsMap) {
    const resultsSection = document.getElementById('results-section');
    
    for (const [modelName, mcqs] of Object.entries(resultsMap)) {
        const modelDiv = document.createElement('div');
        modelDiv.className = 'model-result glass-panel';
        
        const title = document.createElement('h2');
        title.className = 'model-title';
        title.textContent = modelName.replace('_', ' ');
        modelDiv.appendChild(title);

        if (!mcqs || mcqs.length === 0) {
            const p = document.createElement('p');
            p.textContent = "No questions generated.";
            modelDiv.appendChild(p);
        } else {
            mcqs.forEach((mcq, idx) => {
                if(mcq.error) {
                    const p = document.createElement('p');
                    p.style.color = "#ef4444";
                    p.textContent = `Error: ${mcq.error}`;
                    modelDiv.appendChild(p);
                    return;
                }

                const card = document.createElement('div');
                card.className = 'mcq-card';

                const qText = document.createElement('div');
                qText.className = 'mcq-question';
                qText.textContent = `Q${idx + 1}: ${mcq.question}`;
                card.appendChild(qText);

                const optionsList = document.createElement('ul');
                optionsList.className = 'mcq-options';

                const letters = ['A', 'B', 'C', 'D'];
                (mcq.options || []).forEach((opt, oIdx) => {
                    const li = document.createElement('li');
                    li.className = 'mcq-option';
                    if (opt === mcq.correct_answer) {
                        li.classList.add('correct');
                    }

                    const spanL = document.createElement('span');
                    spanL.className = 'option-letter';
                    spanL.textContent = letters[oIdx] || '-';

                    const spanT = document.createElement('span');
                    spanT.textContent = opt;

                    li.appendChild(spanL);
                    li.appendChild(spanT);
                    optionsList.appendChild(li);
                });

                card.appendChild(optionsList);
                modelDiv.appendChild(card);
            });
        }
        
        resultsSection.appendChild(modelDiv);
    }
}
