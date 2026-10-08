"""Realistic Enterprise Meeting Scenario Generator."""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class MeetingScenario:
    """A complete structured meeting scenario with transcripts, speakers, and annotations."""
    scenario_id: str
    title: str
    scenario_type: str
    speakers: List[str]
    turns: List[Dict[str, Any]]
    ground_truth_summary: str
    ground_truth_decisions: List[str]
    ground_truth_action_items: List[Dict[str, str]]
    ground_truth_topics: List[str]


def generate_sprint_planning() -> MeetingScenario:
    """Sprint Planning Meeting for Cloud Platform."""
    speakers = ["Alex (Scrum Master)", "Maya (Backend Lead)", "Leo (Frontend Dev)", "Priya (QA Engineer)"]
    turns = [
        {"speaker": "Alex (Scrum Master)", "start_sec": 0.0, "end_sec": 4.5, "text": "Good morning team, let's kick off Sprint 42 planning. Today our focus is payment gateway migration and real-time alerts."},
        {"speaker": "Maya (Backend Lead)", "start_sec": 5.0, "end_sec": 11.2, "text": "On backend, the Stripe v3 upgrade is almost complete. However, we noticed intermittent timeout errors when testing webhook retries under high concurrency."},
        {"speaker": "Leo (Frontend Dev)", "start_sec": 12.0, "end_sec": 17.8, "text": "From UI side, the new checkout modal is built, but we need updated endpoint contracts from Maya before connecting live tokens."},
        {"speaker": "Alex (Scrum Master)", "start_sec": 18.5, "end_sec": 23.0, "text": "Understood. Maya, can you finalize the OpenAPI specification for checkout by tomorrow EOD?"},
        {"speaker": "Maya (Backend Lead)", "start_sec": 23.5, "end_sec": 28.0, "text": "Yes, I will finalize the OpenAPI schema and fix the webhook retry logic by tomorrow EOD."},
        {"speaker": "Priya (QA Engineer)", "start_sec": 29.0, "end_sec": 35.5, "text": "I will prepare end-to-end integration test suites for the payment flow by Friday so we can run load testing over the weekend."},
        {"speaker": "Leo (Frontend Dev)", "start_sec": 36.0, "end_sec": 41.5, "text": "Great, then I will hook up the checkout frontend components to the staging API by Thursday."},
        {"speaker": "Alex (Scrum Master)", "start_sec": 42.0, "end_sec": 48.0, "text": "Awesome. So we decided to proceed with Stripe v3 cutover next Tuesday, provided Priya's load tests pass with zero regression."}
    ]
    summary = "Sprint 42 focused on completing the Stripe v3 payment gateway migration and checkout modal. Maya committed to delivering OpenAPI specifications and webhook retry fixes by tomorrow EOD. Leo will connect frontend components by Thursday, and Priya will execute end-to-end load testing by Friday. The team agreed on a production cutover target for next Tuesday."
    decisions = [
        "Cutover to Stripe v3 on next Tuesday pending successful load testing",
        "Adopt OpenAPI schema as the single source of truth for checkout contracts"
    ]
    action_items = [
        {"task": "Finalize OpenAPI schema and fix webhook retry logic", "assignee": "Maya", "deadline": "Tomorrow EOD", "priority": "high"},
        {"task": "Prepare end-to-end integration and load test suites", "assignee": "Priya", "deadline": "Friday", "priority": "high"},
        {"task": "Hook up checkout frontend components to staging API", "assignee": "Leo", "deadline": "Thursday", "priority": "medium"}
    ]
    topics = ["Payment Gateway Migration", "API Contract Synchronization", "QA Load Testing & Cutover Plan"]

    return MeetingScenario(
        scenario_id="sprint_planning_42",
        title="Sprint 42 Planning — Stripe v3 Migration",
        scenario_type="Sprint Planning",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_tech_postmortem() -> MeetingScenario:
    """Production Outage Post-Mortem."""
    speakers = ["David (Site Reliability Lead)", "Sarah (Principal Architect)", "Ken (DevOps Engineer)", "Rachel (VP Engineering)"]
    turns = [
        {"speaker": "David (Site Reliability Lead)", "start_sec": 0.0, "end_sec": 6.2, "text": "Let's review the incident from yesterday at 14:15 UTC where the user auth cluster experienced 42 minutes of elevated 504 gateway timeouts."},
        {"speaker": "Sarah (Principal Architect)", "start_sec": 7.0, "end_sec": 14.5, "text": "The root cause was a connection pool leak in the Redis session cache caused by an unhandled TLS handshake retry loop."},
        {"speaker": "Ken (DevOps Engineer)", "start_sec": 15.0, "end_sec": 21.0, "text": "When CPU spiked to 98%, Kubernetes horizontal pod autoscaling failed to trigger because the metric server crashed simultaneously."},
        {"speaker": "Rachel (VP Engineering)", "start_sec": 22.0, "end_sec": 27.5, "text": "That's unacceptable for our enterprise SLA. We need hard circuit breakers and independent alerting."},
        {"speaker": "Sarah (Principal Architect)", "start_sec": 28.0, "end_sec": 34.0, "text": "I will deploy the Redis connection pooling patch with exponential backoff and timeout caps by tonight."},
        {"speaker": "David (Site Reliability Lead)", "start_sec": 35.0, "end_sec": 41.5, "text": "I will configure external Datadog synthetic monitors bypassing in-cluster metrics by tomorrow noon."},
        {"speaker": "Ken (DevOps Engineer)", "start_sec": 42.0, "end_sec": 48.0, "text": "I will update the on-call runbook and run a chaos engineering drill on staging by Friday."},
        {"speaker": "Rachel (VP Engineering)", "start_sec": 49.0, "end_sec": 55.0, "text": "Approved. We agreed to mandate circuit breakers on all external cache dependencies starting immediately."}
    ]
    summary = "The post-mortem analyzed a 42-minute auth service outage caused by a Redis connection pool leak and HPA metrics failure. Sarah committed to patching the Redis connection pooling by tonight, David will set up external synthetic monitors by tomorrow, and Ken will update on-call runbooks and conduct chaos drills by Friday. The team mandated circuit breakers across all cache dependencies."
    decisions = [
        "Mandate circuit breakers on all external cache dependencies immediately",
        "Adopt independent external monitoring to eliminate single-point-of-failure metrics"
    ]
    action_items = [
        {"task": "Deploy Redis connection pooling patch with exponential backoff", "assignee": "Sarah", "deadline": "Tonight", "priority": "urgent"},
        {"task": "Configure external Datadog synthetic monitors", "assignee": "David", "deadline": "Tomorrow noon", "priority": "high"},
        {"task": "Update on-call runbook and run chaos drill on staging", "assignee": "Ken", "deadline": "Friday", "priority": "high"}
    ]
    topics = ["Root Cause Analysis", "Autoscaler Infrastructure Failure", "Preventative Hardening Action Plan"]

    return MeetingScenario(
        scenario_id="tech_postmortem_redis",
        title="P0 Incident Post-Mortem — Auth Cluster 504 Outage",
        scenario_type="Incident Post-Mortem",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_board_strategy() -> MeetingScenario:
    """Executive Board Meeting."""
    speakers = ["Elena (CEO)", "Marcus (CFO)", "Dr. Aris (CTO)", "Chloe (Chief Product Officer)"]
    turns = [
        {"speaker": "Elena (CEO)", "start_sec": 0.0, "end_sec": 5.0, "text": "Welcome everyone. Today we are reviewing our Q3 performance and authorizing the FY27 GenAI enterprise roadmap."},
        {"speaker": "Marcus (CFO)", "start_sec": 5.8, "end_sec": 12.0, "text": "Q3 closed at 18.4 million ARR, beating forecast by 12 percent. Operating cash flow is positive at 3.2 million."},
        {"speaker": "Dr. Aris (CTO)", "start_sec": 13.0, "end_sec": 19.5, "text": "To maintain our competitive moat, we need a 2.5 million capital investment in dedicated GPU training clusters and on-premise model serving."},
        {"speaker": "Chloe (Chief Product Officer)", "start_sec": 20.2, "end_sec": 26.5, "text": "Enterprise clients in financial services and healthcare won't adopt our platform unless we support sovereign private VPC deployments."},
        {"speaker": "Elena (CEO)", "start_sec": 27.0, "end_sec": 32.5, "text": "I agree. The market window is open now. Marcus, can we allocate the 2.5 million from retained earnings?"},
        {"speaker": "Marcus (CFO)", "start_sec": 33.0, "end_sec": 38.0, "text": "Yes, I will prepare the revised FY27 capital allocation budget reflecting the 2.5 million GPU investment by next Monday."},
        {"speaker": "Dr. Aris (CTO)", "start_sec": 39.0, "end_sec": 44.5, "text": "I will deliver the vendor benchmark for NVIDIA H100 versus B200 clusters by end of month."},
        {"speaker": "Elena (CEO)", "start_sec": 45.0, "end_sec": 51.0, "text": "Resolution passed: the board unanimously approved the 2.5 million investment for private AI infrastructure."}
    ]
    summary = "The board reviewed Q3 performance ($18.4M ARR, +12% over forecast) and unanimously approved a $2.5M capital allocation for private GPU clusters and enterprise sovereign AI deployments. Marcus will prepare the revised budget by next Monday, while Dr. Aris will finalize GPU hardware benchmarks by end of month."
    decisions = [
        "Unanimously approved $2.5M capital allocation for private enterprise AI infrastructure",
        "Prioritize private sovereign VPC deployments for healthcare and finance clients"
    ]
    action_items = [
        {"task": "Prepare revised FY27 capital budget with 2.5M GPU allocation", "assignee": "Marcus", "deadline": "Next Monday", "priority": "high"},
        {"task": "Deliver vendor benchmark for NVIDIA H100 vs B200 clusters", "assignee": "Dr. Aris", "deadline": "End of month", "priority": "medium"}
    ]
    topics = ["Q3 Financial Performance", "Enterprise AI & Sovereign Infrastructure", "Capital Allocation Approval"]

    return MeetingScenario(
        scenario_id="board_strategy_q3",
        title="Q3 Executive Board & AI Capital Investment Strategy",
        scenario_type="Executive Board",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_client_discovery() -> MeetingScenario:
    """Client Discovery & Technical Scoping."""
    speakers = ["Sam (Solutions Architect)", "Jessica (Enterprise AE)", "Vikram (Client VP Infrastructure)"]
    turns = [
        {"speaker": "Jessica (Enterprise AE)", "start_sec": 0.0, "end_sec": 5.0, "text": "Thanks Vikram for joining us today. Our goal is to assess your compliance requirements and technical onboarding schedule."},
        {"speaker": "Vikram (Client VP Infrastructure)", "start_sec": 5.8, "end_sec": 13.0, "text": "We operate in heavily regulated banking environments. We require SOC2 Type II compliance, zero data retention for LLM training, and SAML SSO."},
        {"speaker": "Sam (Solutions Architect)", "start_sec": 14.0, "end_sec": 20.0, "text": "Our platform meets SOC2 and ISO27001 out of the box, and we provide customer-managed encryption keys via AWS KMS."},
        {"speaker": "Vikram (Client VP Infrastructure)", "start_sec": 21.0, "end_sec": 27.0, "text": "That's great. If we sign the pilot by October 15, how quickly can our 5,000 employees be provisioned?"},
        {"speaker": "Sam (Solutions Architect)", "start_sec": 28.0, "end_sec": 34.0, "text": "I will deliver the technical architecture whitepaper and Okta SCIM integration guide by Wednesday."},
        {"speaker": "Jessica (Enterprise AE)", "start_sec": 35.0, "end_sec": 40.5, "text": "I will send over the customized enterprise master service agreement and pilot pricing tier by tomorrow afternoon."},
        {"speaker": "Vikram (Client VP Infrastructure)", "start_sec": 41.0, "end_sec": 46.0, "text": "Sounds good. We agreed to review the MSA with our legal team on Thursday and finalize the pilot date."}
    ]
    summary = "The discovery call focused on compliance, Okta SCIM provisioning, and AWS KMS encryption for banking deployment. Sam will deliver the architecture whitepaper and SCIM integration guide by Wednesday. Jessica will share the enterprise MSA and pilot pricing by tomorrow afternoon. The client agreed to legal review on Thursday with an October 15 target pilot."
    decisions = [
        "Include customer-managed encryption keys (AWS KMS) in standard pilot tier",
        "Target October 15 for 5,000-seat employee onboarding pilot"
    ]
    action_items = [
        {"task": "Deliver technical architecture whitepaper and Okta SCIM guide", "assignee": "Sam", "deadline": "Wednesday", "priority": "high"},
        {"task": "Send enterprise MSA contract and pilot pricing tier", "assignee": "Jessica", "deadline": "Tomorrow afternoon", "priority": "high"},
        {"task": "Review MSA with corporate legal team", "assignee": "Vikram", "deadline": "Thursday", "priority": "medium"}
    ]
    topics = ["Banking Regulatory & Compliance Requirements", "SCIM Identity & Encryption Architecture", "Pilot Commercial Agreement"]

    return MeetingScenario(
        scenario_id="client_discovery_fintech",
        title="Enterprise Client Technical Discovery — FinTech Onboarding",
        scenario_type="Client Discovery",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_multilingual_india_sync() -> MeetingScenario:
    """Multilingual Engineering Sync (Hindi, Telugu, and English)."""
    speakers = ["Rajesh (Engineering Manager)", "Sneha (Backend Lead)", "Karthik (Mobile & ML Lead)", "Ananya (Product Manager)"]
    turns = [
        {
            "speaker": "Rajesh (Engineering Manager)",
            "start_sec": 0.0,
            "end_sec": 6.5,
            "language": "en",
            "text": "Good morning everyone. Let's do our weekly multi-region sprint sync. Sneha, payment gateway latency pe kya update hai?",
        },
        {
            "speaker": "Sneha (Backend Lead)",
            "start_sec": 7.0,
            "end_sec": 14.2,
            "language": "hi",
            "translation": "Redis cache has reduced latency by 40%. I will deploy the database index optimization patch by tomorrow evening.",
            "text": "हाँ राजेश, Redis cache लगाने के बाद latency 40% कम हो गई है। मैं कल शाम तक database index optimization patch deploy करूँगी।",
        },
        {
            "speaker": "Karthik (Mobile & ML Lead)",
            "start_sec": 15.0,
            "end_sec": 22.8,
            "language": "te",
            "translation": "The offline speech recognition model on the mobile client is working smoothly. I will share the Android SDK build with the QA team by Friday.",
            "text": "మొబైల్ క్లయింట్‌లో offline speech recognition model చాలా వేగంగా పనిచేస్తోంది. నేను శుక్రవారం లోగా Android SDK build ని QA టీమ్‌కి పంపిస్తాను.",
        },
        {
            "speaker": "Ananya (Product Manager)",
            "start_sec": 23.5,
            "end_sec": 30.5,
            "language": "hi",
            "translation": "Excellent. We also need to localize the dashboard UI for both Hindi and Telugu before next Monday.",
            "text": "बहुत बढ़िया! We also need to localize the dashboard UI for both Hindi and Telugu before next Monday so regional pilot users can test it.",
        },
        {
            "speaker": "Karthik (Mobile & ML Lead)",
            "start_sec": 31.0,
            "end_sec": 36.5,
            "language": "te",
            "translation": "Sure, I will complete the Telugu string localization by Thursday evening.",
            "text": "తప్పకుండా, నేను గురువారం సాయంత్రం లోగా Telugu localization strings ని finalize చేసి PR raise చేస్తాను.",
        },
        {
            "speaker": "Sneha (Backend Lead)",
            "start_sec": 37.0,
            "end_sec": 42.0,
            "language": "hi",
            "translation": "And I will finalize the Hindi localization translations by Friday morning.",
            "text": "और मैं शुक्रवार सुबह तक Hindi translations review करके master branch में merge करवा दूँगी।",
        },
        {
            "speaker": "Rajesh (Engineering Manager)",
            "start_sec": 42.5,
            "end_sec": 49.0,
            "language": "en",
            "text": "Fantastic work team. So we decided to roll out the multilingual pilot in Mumbai and Hyderabad next Tuesday.",
        },
    ]
    summary = "The multilingual engineering sync reviewed payment gateway latency, mobile offline speech models, and regional localization. Sneha confirmed 40% latency reduction via Redis and committed to deploying database index optimizations by tomorrow evening. Karthik reported offline speech recognition success on mobile and committed to sharing the Android SDK by Friday and Telugu strings by Thursday. Sneha will finalize Hindi translations by Friday. The team agreed on rolling out the multilingual pilot in Mumbai and Hyderabad next Tuesday."
    decisions = [
        "Roll out multilingual meeting pilot in Mumbai and Hyderabad next Tuesday",
        "Mandate Hindi and Telugu localizations before regional beta release",
    ]
    action_items = [
        {"task": "Deploy database index optimization patch", "assignee": "Sneha", "deadline": "कल शाम तक", "priority": "high"},
        {"task": "Share Android SDK build with QA team", "assignee": "Karthik", "deadline": "శుక్రవారం లోగా", "priority": "high"},
        {"task": "Finalize Telugu localization strings and raise PR", "assignee": "Karthik", "deadline": "గురువారం సాయంత్రం", "priority": "medium"},
        {"task": "Review and merge Hindi localization translations", "assignee": "Sneha", "deadline": "शुक्रवार सुबह तक", "priority": "medium"},
    ]
    topics = ["Payment Gateway Latency & DB Optimization", "Mobile Offline Speech Recognition SDK", "Regional Localization (Hindi & Telugu)", "Mumbai & Hyderabad Pilot Rollout"]

    return MeetingScenario(
        scenario_id="multilingual_india_sync",
        title="Multilingual Engineering Sync — Hindi, Telugu & English",
        scenario_type="Multilingual Engineering Sync",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_long_executive_townhall() -> MeetingScenario:
    """Long-duration Executive Town Hall & Strategic Review (24 turns across 5 executive leaders)."""
    speakers = [
        "Sarah (CEO)",
        "Marcus (VP Engineering)",
        "Elena (Head of Product)",
        "David (Chief Revenue Officer)",
        "Chloe (VP Operations)",
    ]
    turns = [
        {"speaker": "Sarah (CEO)", "start_sec": 0.0, "end_sec": 8.0, "text": "Welcome everyone to our extended Q4 strategic roadmap alignment. Today we will evaluate our infrastructure scalability, enterprise pipeline, and operational budgeting."},
        {"speaker": "David (Chief Revenue Officer)", "start_sec": 8.5, "end_sec": 17.0, "text": "Starting with commercial growth, our enterprise ARR grew 38% this quarter. However, prospective healthcare customers are blocked waiting on HIPAA compliance certification."},
        {"speaker": "Chloe (VP Operations)", "start_sec": 17.5, "end_sec": 26.0, "text": "Regarding compliance, the external audit firm completed preliminary penetration testing last week. I will deliver the consolidated HIPAA readiness report and budget audit by Friday EOD."},
        {"speaker": "Sarah (CEO)", "start_sec": 26.5, "end_sec": 32.0, "text": "Excellent Chloe. Moving on to engineering architecture, Marcus, how is our multi-region latency holding up?"},
        {"speaker": "Marcus (VP Engineering)", "start_sec": 32.5, "end_sec": 42.0, "text": "Our European cluster is performing within SLA, but database connection spikes during peak US morning traffic are causing 2% queue saturation."},
        {"speaker": "Elena (Head of Product)", "start_sec": 42.5, "end_sec": 51.0, "text": "Users are also requesting background meeting transcription with speaker search. We need to prioritize real-time diarization in the web client."},
        {"speaker": "Marcus (VP Engineering)", "start_sec": 51.5, "end_sec": 60.5, "text": "I am going to migrate our primary search cluster to OpenSearch and deploy connection pool limits by next Tuesday."},
        {"speaker": "Elena (Head of Product)", "start_sec": 61.0, "end_sec": 69.5, "text": "From product side, I'll handle the enterprise customer onboarding survey analysis and deliver UI wireframes by tomorrow afternoon."},
        {"speaker": "David (Chief Revenue Officer)", "start_sec": 70.0, "end_sec": 79.0, "text": "If Elena delivers the prototypes this week, I will close the three Fortune 500 pilots before next month."},
        {"speaker": "Sarah (CEO)", "start_sec": 79.5, "end_sec": 88.0, "text": "That is a huge milestone. We decided to increase our R&D cloud budget by 25% starting next sprint to fund the OpenSearch cluster."},
        {"speaker": "Chloe (VP Operations)", "start_sec": 88.5, "end_sec": 96.0, "text": "I will coordinate with finance to approve the updated cloud allocation before Thursday morning."},
        {"speaker": "Marcus (VP Engineering)", "start_sec": 96.5, "end_sec": 105.0, "text": "Regarding contractor access, I will enforce hardware security keys across all staging and production environments by next Monday."},
        {"speaker": "Sarah (CEO)", "start_sec": 105.5, "end_sec": 113.0, "text": "Agreed. We agreed to mandate SOC2 Type II compliance audits across all contractors and third-party vendors."},
        {"speaker": "Elena (Head of Product)", "start_sec": 113.5, "end_sec": 121.0, "text": "On telemetry, we should also track participant talk-time inequality metrics directly in our customer dashboard."},
        {"speaker": "Marcus (VP Engineering)", "start_sec": 121.5, "end_sec": 129.0, "text": "That will be straightforward once our analytics event pipeline is streaming. I'll take the lead on the event schema by Wednesday."},
        {"speaker": "David (Chief Revenue Officer)", "start_sec": 129.5, "end_sec": 137.0, "text": "Sales engineering team will need customer-facing one-pagers explaining our deep learning architecture and privacy guarantees."},
        {"speaker": "Elena (Head of Product)", "start_sec": 137.5, "end_sec": 145.0, "text": "I will prepare the enterprise security architecture collateral and publish it to the sales portal by Thursday EOD."},
        {"speaker": "Chloe (VP Operations)", "start_sec": 145.5, "end_sec": 152.0, "text": "I will schedule our quarterly executive offsite and send calendar invites to all department heads by tomorrow."},
        {"speaker": "Sarah (CEO)", "start_sec": 152.5, "end_sec": 160.0, "text": "Regarding hiring, we decided to open four senior deep learning positions in North America and EMEA."},
        {"speaker": "Chloe (VP Operations)", "start_sec": 160.5, "end_sec": 167.0, "text": "I will post the four senior ML engineering job requisitions on LinkedIn and Greenhouse by Friday noon."},
        {"speaker": "Marcus (VP Engineering)", "start_sec": 167.5, "end_sec": 174.0, "text": "I will draft the technical interview assessment rubrics by Monday morning."},
        {"speaker": "David (Chief Revenue Officer)", "start_sec": 174.5, "end_sec": 181.0, "text": "I will update the revenue forecasts reflecting the increased headcount and share them with the board by Wednesday."},
        {"speaker": "Elena (Head of Product)", "start_sec": 181.5, "end_sec": 187.0, "text": "Looking forward to executing on the product milestones."},
        {"speaker": "Sarah (CEO)", "start_sec": 187.5, "end_sec": 195.0, "text": "Thank you everyone for the alignment. We have clear ownership, strong momentum, and a shared vision for scale."},
    ]
    summary = "The Executive Town Hall reviewed commercial expansion, engineering infrastructure, product AI features, and operational hiring. David noted 38% ARR growth, while Chloe committed to delivering the HIPAA readiness report by Friday EOD. Marcus will migrate search to OpenSearch by next Tuesday and enforce hardware keys by Monday. Elena committed to UI wireframes by tomorrow afternoon and security collateral by Thursday EOD. The team approved a 25% R&D cloud budget increase, mandated contractor SOC2 compliance, and decided to hire four senior deep learning engineers."
    decisions = [
        "Increase R&D cloud infrastructure budget by 25% starting next sprint",
        "Mandate SOC2 Type II compliance audits across all contractors and vendors",
        "Open four senior deep learning engineering requisitions across North America and EMEA",
    ]
    action_items = [
        {"task": "Deliver consolidated HIPAA readiness report and budget audit", "assignee": "Chloe", "deadline": "Friday EOD", "priority": "high"},
        {"task": "Migrate primary search cluster to OpenSearch and deploy connection limits", "assignee": "Marcus", "deadline": "Next Tuesday", "priority": "high"},
        {"task": "Deliver UI wireframes and customer survey analysis", "assignee": "Elena", "deadline": "Tomorrow afternoon", "priority": "medium"},
        {"task": "Enforce hardware security keys across staging and production", "assignee": "Marcus", "deadline": "Next Monday", "priority": "urgent"},
        {"task": "Prepare enterprise security collateral for sales portal", "assignee": "Elena", "deadline": "Thursday EOD", "priority": "medium"},
        {"task": "Post four senior ML job requisitions on LinkedIn and Greenhouse", "assignee": "Chloe", "deadline": "Friday noon", "priority": "medium"},
    ]
    topics = [
        "Commercial Growth & HIPAA Compliance",
        "Cloud Infrastructure & OpenSearch Migration",
        "Product Telemetry & Analytics Dashboard",
        "Hiring & Headcount Expansion",
    ]

    return MeetingScenario(
        scenario_id="long_executive_townhall",
        title="Extended Executive Town Hall — Strategy, Infrastructure & Hiring",
        scenario_type="Executive Town Hall",
        speakers=speakers,
        turns=turns,
        ground_truth_summary=summary,
        ground_truth_decisions=decisions,
        ground_truth_action_items=action_items,
        ground_truth_topics=topics,
    )


def generate_meeting_scenarios() -> Dict[str, MeetingScenario]:
    """Generates complete dictionary of pre-packaged meeting scenarios."""
    scenarios = {
        "sprint_planning": generate_sprint_planning(),
        "tech_postmortem": generate_tech_postmortem(),
        "board_strategy": generate_board_strategy(),
        "client_discovery": generate_client_discovery(),
        "multilingual_india_sync": generate_multilingual_india_sync(),
        "long_executive_townhall": generate_long_executive_townhall(),
    }
    return scenarios
