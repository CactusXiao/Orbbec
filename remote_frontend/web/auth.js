// Account UI: credentials never enter URLs, IndexedDB, or local/session storage.
export function accountUI({ api, suspend, resume, notice }) {
  const $ = (id) => document.getElementById(id);
  let user = null,
    activityAt = 0,
    checking = false,
    editing = null;
  async function signedOut(message = "请使用管理员分配的账号登录。") {
    user = null;
    await suspend();
    for (const d of document.querySelectorAll("dialog[open]")) d.close();
    $("loginPanel").hidden = false;
    $("accountTools").hidden = true;
    $("connection").textContent = "未登录";
    $("loginError").textContent = message;
  }
  async function accept(identity) {
    user = identity;
    $("loginPanel").hidden = true;
    $("accountTools").hidden = false;
    $("accountName").textContent = identity.username;
    $("changePassword").hidden = identity.auth_source === "backend";
    $("manageAccounts").hidden = !identity.admin || identity.must_change;
    $("connection").textContent = identity.must_change
      ? "请更换初始密码"
      : "已登录";
    $("passwordCancel").hidden = identity.must_change;
    if (identity.must_change) {
      $("passwordNote").textContent =
        "首次登录或管理员重置后，请先设置自己的密码。更换后需重新登录。";
      $("passwordDialog").showModal();
    } else await resume(identity);
  }
  $("loginForm").onsubmit = async (e) => {
    e.preventDefault();
    const button = $("loginSubmit");
    button.disabled = true;
    $("loginError").textContent = "正在验证…";
    try {
      const identity = await api("/api/login", {
        username: $("loginUsername").value,
        password: $("loginPassword").value,
      });
      $("loginPassword").value = "";
      $("loginError").textContent = "";
      await accept(identity);
    } catch (e) {
      $("loginError").textContent = e.message;
    } finally {
      button.disabled = false;
    }
  };
  $("logout").onclick = async () => {
    try {
      await suspend();
      await api("/api/logout", {});
      await signedOut(
        "已退出。未提交的草稿仍保留在这个浏览器，仅在登录原账号后显示。",
      );
    } catch (e) {
      notice("退出未完成：" + e.message);
    }
  };
  $("changePassword").onclick = () => {
    $("passwordNote").textContent = "更换密码会退出该账号的所有浏览器。";
    $("passwordDialog").showModal();
  };
  $("passwordLogout").onclick = () => $("logout").onclick();
  $("passwordCancel").onclick = () => $("passwordDialog").close();
  $("passwordDialog").addEventListener("cancel", (e) => {
    if (user?.must_change) e.preventDefault();
  });
  $("passwordForm").onsubmit = async (e) => {
    e.preventDefault();
    $("passwordError").textContent = "";
    if ($("newPassword").value !== $("repeatPassword").value) {
      $("passwordError").textContent = "两次新密码不一致";
      return;
    }
    $("passwordSubmit").disabled = true;
    try {
      await api("/api/password", {
        old_password: $("oldPassword").value,
        new_password: $("newPassword").value,
      });
      $("passwordForm").reset();
      $("passwordDialog").close();
      await signedOut("密码已更新，请使用新密码登录。");
    } catch (e) {
      $("passwordError").textContent = e.message;
    } finally {
      $("passwordSubmit").disabled = false;
    }
  };
  function resetEditor() {
    editing = null;
    $("accountForm").reset();
    $("newUsername").readOnly = false;
    $("temporaryPassword").required = true;
    $("temporaryPassword").disabled = false;
    $("backendAccount").disabled = false;
    $("accountSave").textContent = "创建账号";
    $("accountError").textContent = "";
    $("accountEnabled").checked = true;
  }
  async function loadAccounts() {
    const [accounts, tasks] = await Promise.all([
      api("/api/accounts"),
      api("/api/account-tasks"),
    ]);
    $("allowedTasks").replaceChildren(...tasks.map((t) => new Option(t, t)));
    $("accountList").replaceChildren();
    for (const a of accounts) {
      const row = document.createElement("div");
      row.className = "row";
      const label = document.createElement("span");
      label.textContent = `${a.username} · ${a.auth_source === "backend" ? "原后端账号 · " : ""}${a.admin ? "管理员" : a.roles.join(" / ").toUpperCase()} · ${a.enabled ? "启用" : "停用"} · ${a.tasks.includes("*") ? "全部任务" : a.tasks.join("、") || "未分配任务"}`;
      row.append(label);
      if (!a.admin) {
        const edit = document.createElement("button");
        edit.textContent =
          a.auth_source === "backend" ? "编辑外包权限" : "编辑 / 重置密码";
        edit.onclick = () => {
          resetEditor();
          editing = a.username;
          $("newUsername").value = a.username;
          $("newUsername").readOnly = true;
          $("temporaryPassword").required = false;
          $("backendAccount").checked = a.auth_source === "backend";
          $("backendAccount").disabled = true;
          $("temporaryPassword").disabled = a.auth_source === "backend";
          $("allowLabel").checked = a.roles.includes("label");
          $("allowQC").checked = a.roles.includes("qc");
          $("allTasks").checked = a.tasks.includes("*");
          for (const option of $("allowedTasks").options)
            option.selected = a.tasks.includes(option.value);
          $("accountEnabled").checked = a.enabled;
          $("accountSave").textContent = "保存修改并使旧登录失效";
        };
        row.append(edit);
      }
      $("accountList").append(row);
    }
  }
  $("manageAccounts").onclick = async () => {
    try {
      resetEditor();
      await loadAccounts();
      $("accountsDialog").showModal();
    } catch (e) {
      notice(e.message);
    }
  };
  $("closeAccounts").onclick = () => {
    $("accountForm").reset();
    $("accountsDialog").close();
  };
  $("newAccount").onclick = resetEditor;
  $("backendAccount").onchange = () => {
    const backend = $("backendAccount").checked;
    $("temporaryPassword").disabled = backend;
    $("temporaryPassword").required = !backend && !editing;
    $("temporaryPassword").value = "";
    $("accountSave").textContent = backend
      ? "关联账号并分配外包权限"
      : "创建账号";
  };
  $("accountForm").onsubmit = async (e) => {
    e.preventDefault();
    $("accountSave").disabled = true;
    const roles = [
      $("allowLabel").checked ? "label" : null,
      $("allowQC").checked ? "qc" : null,
    ].filter(Boolean);
    const tasks = $("allTasks").checked
      ? ["*"]
      : [...$("allowedTasks").selectedOptions].map((o) => o.value);
    const body = {
      username: $("newUsername").value,
      roles,
      tasks,
      enabled: $("accountEnabled").checked,
      auth_source: $("backendAccount").checked ? "backend" : "local",
    };
    if ($("temporaryPassword").value)
      body.password = $("temporaryPassword").value;
    try {
      await api(
        editing
          ? `/api/accounts/${encodeURIComponent(editing)}`
          : "/api/accounts",
        body,
      );
      resetEditor();
      await loadAccounts();
      $("accountError").textContent =
        body.auth_source === "backend"
          ? "已关联原后端账号，请使用原账号密码登录，无需重复注册。"
          : "已保存。请通过企业的安全渠道交付账号和初始密码；用户首次登录时必须改密。";
    } catch (e) {
      $("accountError").textContent = e.message;
    } finally {
      $("accountSave").disabled = false;
    }
  };
  for (const event of ["pointerdown", "keydown", "wheel"])
    window.addEventListener(
      event,
      () => {
        if (!user || Date.now() - activityAt < 60000) return;
        activityAt = Date.now();
        api("/api/activity", {}).catch(() => {});
      },
      { passive: true },
    );
  setInterval(async () => {
    if (!user || checking) return;
    checking = true;
    try {
      await api("/api/identity");
    } catch {
    } finally {
      checking = false;
    }
  }, 60000);
  return {
    async start() {
      // Old shared keys are retired; never exchange them for an account session.
      if (location.hash)
        history.replaceState(null, "", location.pathname + location.search);
      try {
        await accept(await api("/api/identity"));
      } catch (e) {
        await signedOut(
          e.status === 401
            ? "请使用自己的账号和密码登录。"
            : "暂时无法连接后端，请检查 VPN 后重试。",
        );
      }
    },
    expired(message) {
      if (user) signedOut(message).catch((e) => notice(e.message));
    },
  };
}
